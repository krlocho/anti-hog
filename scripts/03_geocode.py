"""
Geocodifica los 'lugar' de events.csv contra un gazetteer de Lanzarote.

Genera DOS esquemas de zona:
  - zona   : 6 zonas amplias (capital, sur, yaiza, interior, centro, norte)
             → se conserva para el documento/histórico, RAG y comparativas.
  - zona12 : 12 zonas finas (municipios + núcleos calientes)
             → es la que usa el modelo predictivo.

Salida: data/processed/events_geocoded.csv  (con columnas zona y zona12)

Uso:
    python scripts/03_geocode.py
"""

import re
import unicodedata
from pathlib import Path

import pandas as pd
from rapidfuzz import fuzz, process

ROOT = Path(__file__).resolve().parents[1]
IN_PATH = ROOT / "data" / "processed" / "events.csv"
OUT_PATH = ROOT / "data" / "processed" / "events_geocoded.csv"

# Gazetteer: (nombre canónico, lat, lon, zona6, zona12)
GAZETTEER = [
    # ── CAPITAL (zona6) ──
    ("arrecife",          28.9630, -13.5476, "capital", "arrecife"),
    ("puerto naos",       28.9614, -13.5305, "capital", "arrecife"),
    ("puerto los marmoles", 28.9700, -13.5283, "capital", "arrecife"),
    ("argana",            28.9700, -13.5750, "capital", "arrecife"),
    ("rambla medular",    28.9630, -13.5476, "capital", "arrecife"),
    ("circunvalacion",    28.9630, -13.5476, "capital", "arrecife"),
    ("montaña mina",      28.9450, -13.5950, "capital", "arrecife"),
    ("hospital",          28.9510, -13.5980, "capital", "arrecife"),
    ("rotonda hospital",  28.9510, -13.5980, "capital", "arrecife"),
    ("biosfera",          28.9620, -13.5510, "capital", "arrecife"),
    ("intermares",        28.9550, -13.5510, "capital", "arrecife"),
    ("cabrera felipe",    28.9650, -13.5550, "capital", "arrecife"),
    ("juan grande",       28.9750, -13.5800, "capital", "arrecife"),
    ("playa honda",       28.9293, -13.6098, "capital", "playa_honda"),
    ("aeropuerto",        28.9445, -13.6044, "capital", "playa_honda"),
    ("guacimeta",         28.9445, -13.6044, "capital", "playa_honda"),
    ("san bartolome",     28.9839, -13.6181, "capital", "san_bartolome"),
    ("guime",             28.9530, -13.5916, "capital", "san_bartolome"),
    ("mozaga",            28.9925, -13.6200, "capital", "san_bartolome"),
    ("cruce mozaga",      28.9925, -13.6200, "capital", "san_bartolome"),
    ("indelasa",          28.9580, -13.6020, "capital", "san_bartolome"),
    ("tahiche",           28.9933, -13.5640, "capital", "tahiche"),
    ("cruce tahiche",     28.9933, -13.5640, "capital", "tahiche"),
    ("malecita",          28.9933, -13.5640, "capital", "tahiche"),
    ("gallo loco",        28.9933, -13.5640, "capital", "tahiche"),

    # ── SUR (zona6) ──
    ("tias",              28.9588, -13.6480, "sur", "tias"),
    ("macher",            28.9540, -13.6790, "sur", "tias"),
    ("la asomada",        28.9665, -13.6921, "sur", "tias"),
    ("conil",             28.9633, -13.6633, "sur", "tias"),
    ("el toro",           28.9550, -13.6450, "sur", "tias"),
    ("puerto del carmen", 28.9219, -13.6671, "sur", "puerto_carmen"),
    ("pto del carmen",    28.9219, -13.6671, "sur", "puerto_carmen"),
    ("matagorda",         28.9281, -13.6470, "sur", "puerto_carmen"),
    ("rancho texas",      28.9300, -13.6700, "sur", "puerto_carmen"),
    ("puerto calero",     28.9180, -13.7028, "sur", "puerto_carmen"),

    # ── YAIZA (zona6) ──
    ("yaiza",             28.9569, -13.7693, "yaiza", "yaiza"),
    ("playa quemada",     28.9000, -13.7472, "yaiza", "yaiza"),
    ("el golfo",          28.9776, -13.8281, "yaiza", "yaiza"),
    ("uga",               28.9586, -13.7464, "yaiza", "yaiza"),
    ("femes",             28.8830, -13.7800, "yaiza", "yaiza"),
    ("los camellos",      29.0050, -13.7400, "yaiza", "yaiza"),
    ("playa blanca",      28.8607, -13.8260, "yaiza", "playa_blanca"),

    # ── INTERIOR (zona6) ──
    ("tinajo",            29.0666, -13.6755, "interior", "tinajo"),
    ("mancha blanca",     29.0428, -13.6779, "interior", "tinajo"),
    ("la geria",          28.9778, -13.7166, "interior", "tinajo"),
    ("masdache",          28.9892, -13.6864, "interior", "tinajo"),
    ("la santa",          29.1167, -13.6660, "interior", "tinajo"),
    ("el cuchillo",       29.0950, -13.6800, "interior", "tinajo"),
    ("vega de tegoyo",    29.0050, -13.6400, "interior", "tinajo"),
    ("cruce tinajo",      29.0666, -13.6755, "interior", "tinajo"),

    # ── CENTRO (zona6) ──
    ("teguise",           29.0593, -13.5641, "centro", "teguise"),
    ("tao",               29.0269, -13.6173, "centro", "teguise"),
    ("tiagua",            29.0335, -13.6325, "centro", "teguise"),
    ("soo",               29.0833, -13.6500, "centro", "teguise"),
    ("munique",           29.0490, -13.6300, "centro", "teguise"),
    ("muniique",          29.0490, -13.6300, "centro", "teguise"),
    ("famara",            29.1167, -13.5667, "centro", "teguise"),
    ("caleta de famara",  29.1167, -13.5667, "centro", "teguise"),
    ("nazaret",           29.0166, -13.5500, "centro", "teguise"),
    ("costa teguise",     28.9990, -13.5181, "centro", "costa_teguise"),
    ("las caletas",       29.1100, -13.4633, "centro", "costa_teguise"),
    ("cruce caleta",      29.1100, -13.4633, "centro", "costa_teguise"),

    # ── NORTE (zona6) ──
    ("haria",             29.1463, -13.4988, "norte", "haria"),
    ("orzola",            29.2186, -13.4506, "norte", "haria"),
    ("punta mujeres",     29.1671, -13.4549, "norte", "haria"),
    ("mala",              29.1095, -13.4710, "norte", "haria"),
    ("arrieta",           29.1409, -13.4636, "norte", "haria"),
    ("guatiza",           29.0913, -13.4750, "norte", "haria"),
    ("tabayesco",         29.1300, -13.4900, "norte", "haria"),
    ("ye",                29.1995, -13.4843, "norte", "haria"),
    ("maguez",            29.1556, -13.4960, "norte", "haria"),
    ("mirador del rio",   29.2167, -13.4833, "norte", "haria"),
]

GAZETTEER_NAMES = [g[0] for g in GAZETTEER]
GAZETTEER_MAP = {g[0]: g for g in GAZETTEER}
FUZZY_THRESHOLD = 75


def normalize(s: str) -> str:
    if not isinstance(s, str):
        return ""
    s = s.lower().strip()
    s = "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c))
    s = re.sub(r"[^\w\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def geocode(lugar: str):
    """Devuelve (canonical, lat, lon, zona6, zona12, score)."""
    if not lugar:
        return (None, None, None, None, None, 0)
    n = normalize(lugar)
    if not n:
        return (None, None, None, None, None, 0)
    # 1. match exacto (el texto contiene un nombre del gazetteer)
    for name in GAZETTEER_NAMES:
        if name in n:
            g = GAZETTEER_MAP[name]
            return (g[0], g[1], g[2], g[3], g[4], 100)
    # 2. fuzzy
    best = process.extractOne(n, GAZETTEER_NAMES, scorer=fuzz.token_set_ratio)
    if best and best[1] >= FUZZY_THRESHOLD:
        g = GAZETTEER_MAP[best[0]]
        return (g[0], g[1], g[2], g[3], g[4], best[1])
    return (None, None, None, None, None, best[1] if best else 0)


def main() -> None:
    df = pd.read_csv(IN_PATH)
    print(f"[geocode] filas leídas: {len(df)}")
    controls = df[df["es_control"] == True].copy()
    print(f"[geocode] eventos de control: {len(controls)}")

    g = controls["lugar"].apply(geocode)
    controls["canonical"] = g.apply(lambda x: x[0])
    controls["lat"]       = g.apply(lambda x: x[1])
    controls["lon"]       = g.apply(lambda x: x[2])
    controls["zona"]      = g.apply(lambda x: x[3])   # 6 zonas
    controls["zona12"]    = g.apply(lambda x: x[4])   # 12 zonas
    controls["geo_score"] = g.apply(lambda x: x[5])

    ok = controls["canonical"].notna().sum()
    pct = ok / len(controls) * 100 if len(controls) else 0
    print(f"[geocode] con coordenadas: {ok} ({pct:.1f}%)\n")
    print("=== zona (6) ===")
    print(controls["zona"].value_counts(dropna=False).to_string())
    print("\n=== zona12 (12) ===")
    print(controls["zona12"].value_counts(dropna=False).to_string())
    print("\n=== top lugares no resueltos ===")
    print(controls[controls["canonical"].isna()]["lugar"].value_counts().head(15).to_string())

    controls.to_csv(OUT_PATH, index=False)
    print(f"\n[geocode] guardado en {OUT_PATH}")


if __name__ == "__main__":
    main()
