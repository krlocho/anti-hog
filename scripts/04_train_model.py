"""
Entrena un modelo XGBoost que predice P(control) para una celda (fecha, zona, slot_3h).

Pipeline:
1. Carga events_geocoded.csv y filtra controles con coordenadas válidas.
2. Construye dataset de "celdas" enumerando (fecha en rango, zona, slot_3h):
   - Positivo (y=1) si en esa celda se observó ≥1 control.
   - Negativo (y=0) en caso contrario.
3. Features: día_semana, mes, año, slot_3h, es_festivo, es_finde, vispera_festivo, zona.
4. Validación temporal: train < cutoff, test ≥ cutoff (último 20% de fechas).
5. Métricas: ROC-AUC, PR-AUC, log_loss + calibración.

Salida:
  models/predictor.pkl  (xgboost + feature schema)
  models/metrics.json   (métricas en test)

Uso:
    python scripts/04_train_model.py [--cutoff-pct 0.8]
"""

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    log_loss,
    roc_auc_score,
)
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[1]
IN_PATH = ROOT / "data" / "processed" / "events_geocoded.csv"
MODEL_PATH = ROOT / "models" / "predictor.pkl"
METRICS_PATH = ROOT / "models" / "metrics.json"
MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

# Festivos Canarias — fallback simple si workalendar falla
try:
    from workalendar.europe.spain import CanaryIslands

    _CAL = CanaryIslands()

    def is_holiday(d: pd.Timestamp) -> bool:
        return _CAL.is_working_day(d.to_pydatetime().date()) is False and d.weekday() < 5

except Exception:

    def is_holiday(d: pd.Timestamp) -> bool:
        return False


SLOT_HOURS = 3  # 8 slots por día: 0-3, 3-6, ..., 21-24


def hora_to_slot(h) -> int | None:
    if pd.isna(h):
        return None
    try:
        hh = int(str(h).split(":")[0])
        return hh // SLOT_HOURS
    except Exception:
        return None


def build_cells(df: pd.DataFrame) -> pd.DataFrame:
    """Genera el dataset de celdas (fecha, zona, slot) con label binaria."""
    df = df.copy()
    df["fecha_evento"] = pd.to_datetime(df["fecha"].fillna(df["pub_date"]), errors="coerce", utc=True).dt.tz_localize(None)
    df = df.dropna(subset=["fecha_evento", "zona"])
    df["slot"] = df["hora"].apply(hora_to_slot)

    # Si no hay hora, asignamos slot=-1 (desconocido) — se agrupará aparte
    df["slot"] = df["slot"].fillna(-1).astype(int)
    df["fecha_dia"] = df["fecha_evento"].dt.normalize()

    # Conjunto de positivos (set para lookup O(1))
    positives = set(zip(df["fecha_dia"], df["zona"], df["slot"]))

    # Detectar período activo: si hay un gap > 90 días sin positivos, conservamos
    # sólo el segmento contiguo más grande (evita meter años muertos en val/test).
    positive_dates = pd.Series(sorted(df["fecha_dia"].unique()))
    gaps = positive_dates.diff()
    if not gaps.empty:
        idx = gaps.idxmax()
        if gaps.iloc[idx] > pd.Timedelta(days=90):
            pre = positive_dates.iloc[:idx]
            post = positive_dates.iloc[idx:]
            if len(pre) >= len(post):
                fmin, fmax = pre.iloc[0], pre.iloc[-1]
            else:
                fmin, fmax = post.iloc[0], post.iloc[-1]
            print(f"[train] gap detectado de {gaps.iloc[idx].days} días → usamos {fmin.date()} → {fmax.date()}")
        else:
            fmin, fmax = positive_dates.iloc[0], positive_dates.iloc[-1]
    else:
        fmin, fmax = df["fecha_dia"].min(), df["fecha_dia"].max()

    all_dates = pd.date_range(fmin, fmax, freq="D")
    all_zonas = sorted(df["zona"].dropna().unique())
    all_slots = list(range(8))  # 0..7 (descartamos slot -1 para celdas negativas)

    print(f"[train] rango fechas usado: {fmin.date()} → {fmax.date()}  días={len(all_dates)}")
    print(f"[train] zonas: {all_zonas}")
    print(f"[train] positivos observados: {len(positives)}")

    rows = []
    for date in all_dates:
        for zona in all_zonas:
            for slot in all_slots:
                y = 1 if (date, zona, slot) in positives else 0
                rows.append((date, zona, slot, y))

    cells = pd.DataFrame(rows, columns=["fecha", "zona", "slot", "y"])
    return cells


def add_features(cells: pd.DataFrame) -> pd.DataFrame:
    cells = cells.copy()
    cells["dia_semana"] = cells["fecha"].dt.dayofweek
    cells["mes"] = cells["fecha"].dt.month
    cells["anio"] = cells["fecha"].dt.year
    cells["es_finde"] = (cells["dia_semana"] >= 5).astype(int)
    cells["es_festivo"] = cells["fecha"].apply(is_holiday).astype(int)
    # Víspera festivo: día_después es festivo
    cells["vispera_festivo"] = cells["fecha"].apply(
        lambda d: is_holiday(d + pd.Timedelta(days=1))
    ).astype(int)
    return cells


def main(cutoff_pct: float) -> None:
    df = pd.read_csv(IN_PATH)
    print(f"[train] eventos cargados: {len(df)}")

    # Sólo eventos con geocoding
    df = df[df["zona"].notna() & (df["es_control"] == True)]
    print(f"[train] eventos con zona válida: {len(df)}")

    cells = build_cells(df)
    cells = add_features(cells)

    print(f"[train] total celdas: {len(cells)}")
    print(f"[train] tasa positivos: {cells['y'].mean():.4f}")

    # Split temporal
    cutoff_date = cells["fecha"].quantile(cutoff_pct, interpolation="nearest")
    train = cells[cells["fecha"] < cutoff_date]
    test = cells[cells["fecha"] >= cutoff_date]
    print(f"[train] cutoff: {cutoff_date.date()}  train={len(train)}  test={len(test)}")

    feature_cols = ["dia_semana", "mes", "anio", "slot", "es_finde", "es_festivo", "vispera_festivo"]
    cat_cols = ["zona"]
    train_X = pd.get_dummies(train[feature_cols + cat_cols], columns=cat_cols)
    test_X = pd.get_dummies(test[feature_cols + cat_cols], columns=cat_cols)
    # alinear columnas
    test_X = test_X.reindex(columns=train_X.columns, fill_value=0)

    train_y = train["y"].values
    test_y = test["y"].values

    pos_weight = (1 - train_y.mean()) / max(train_y.mean(), 1e-6)
    model = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        scale_pos_weight=pos_weight,
        eval_metric="logloss",
        tree_method="hist",
        random_state=42,
    )
    model.fit(train_X, train_y, eval_set=[(test_X, test_y)], verbose=False)

    proba = model.predict_proba(test_X)[:, 1]
    metrics = {
        "roc_auc": float(roc_auc_score(test_y, proba)),
        "pr_auc": float(average_precision_score(test_y, proba)),
        "log_loss": float(log_loss(test_y, proba, labels=[0, 1])),
        "baseline_rate_train": float(train_y.mean()),
        "baseline_rate_test": float(test_y.mean()),
        "n_train": int(len(train)),
        "n_test": int(len(test)),
        "cutoff": str(cutoff_date.date()),
        "features": list(train_X.columns),
    }
    print("\n=== Métricas test ===")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        elif isinstance(v, list):
            continue
        else:
            print(f"  {k}: {v}")

    with open(MODEL_PATH, "wb") as f:
        pickle.dump({"model": model, "features": list(train_X.columns)}, f)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2))

    print(f"\n[train] modelo en  {MODEL_PATH}")
    print(f"[train] métricas en {METRICS_PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cutoff-pct", type=float, default=0.8, help="fracción de fechas para train (resto test)")
    args = parser.parse_args()
    main(args.cutoff_pct)
