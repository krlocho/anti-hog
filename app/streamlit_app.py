"""
Anti Hog — App principal.
Estética: terminal CRT de fósforo ámbar / radar de intercepción.
"""

from __future__ import annotations

import base64
import json
import math
import os
import pickle
import subprocess
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

import folium
import pandas as pd
import plotly.express as px
import streamlit as st
from dotenv import load_dotenv
from streamlit_folium import st_folium

load_dotenv()

ROOT = Path(__file__).resolve().parents[1]
EVENTS_PATH   = ROOT / "data" / "processed" / "events_geocoded.csv"
MODEL_PATH    = ROOT / "models" / "predictor.pkl"
METRICS_PATH  = ROOT / "models" / "metrics.json"
CHROMA_DIR    = ROOT / "rag" / "chroma_db"
SCRAPER       = ROOT / "scripts" / "01_scrape_telegram.py"
LOGO_PATH     = ROOT / "assets" / "logo.png"

# Paleta fósforo ámbar
AMBER   = "#FFB000"
AMBER_2 = "#FF7A00"
DANGER  = "#FF3B30"
SAFE    = "#FFD9A0"
BG      = "#0A0A0B"

LANZAROTE_CENTER = (29.04, -13.58)
ZONA_COORDS = {  # 6 zonas (documento/histórico)
    "capital":  (28.9630, -13.5476),
    "sur":      (28.9588, -13.6480),
    "yaiza":    (28.9000, -13.8000),
    "interior": (29.0500, -13.6800),
    "centro":   (29.0500, -13.5400),
    "norte":    (29.1400, -13.4700),
}
ZONA12_COORDS = {  # 12 zonas finas (modelo)
    "arrecife":      (28.9630, -13.5476),
    "playa_honda":   (28.9293, -13.6098),
    "san_bartolome": (28.9839, -13.6181),
    "tahiche":       (28.9933, -13.5640),
    "costa_teguise": (28.9990, -13.5181),
    "teguise":       (29.0593, -13.5641),
    "tias":          (28.9588, -13.6480),
    "puerto_carmen": (28.9219, -13.6671),
    "yaiza":         (28.9569, -13.7693),
    "playa_blanca":  (28.8607, -13.8260),
    "tinajo":        (29.0666, -13.6755),
    "haria":         (29.1463, -13.4988),
}
SLOT_HOURS = 3

st.set_page_config(
    page_title="ANTI HOG // INTERCEPT",
    page_icon=str(LOGO_PATH) if LOGO_PATH.exists() else "🐷",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ════════════════════════════════════════════════════════════════════════════
# CSS — terminal CRT ámbar
# ════════════════════════════════════════════════════════════════════════════
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Chakra+Petch:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;700;800&display=swap');

:root{
  --amber:#FFB000; --amber2:#FF7A00; --danger:#FF3B30;
  --bg:#0A0A0B; --bg2:#121110; --panel:#15130E;
  --line:rgba(255,176,0,.14); --text:#FFD9A0; --muted:#9A865C;
  --glow:0 0 12px rgba(255,176,0,.45);
}

html,body,[class*="css"]{
  font-family:'JetBrains Mono',monospace !important;
  color:var(--text);
}
.stApp{
  background:
    radial-gradient(120% 80% at 50% -10%, rgba(255,176,0,.06), transparent 60%),
    radial-gradient(140% 120% at 50% 120%, rgba(255,122,0,.05), transparent 55%),
    var(--bg);
}
/* Scanlines + vignette CRT */
.stApp::before{
  content:""; position:fixed; inset:0; pointer-events:none; z-index:9999;
  background:repeating-linear-gradient(0deg, rgba(0,0,0,.18) 0 1px, transparent 1px 3px);
  mix-blend-mode:multiply; opacity:.5;
}
.stApp::after{
  content:""; position:fixed; inset:0; pointer-events:none; z-index:9998;
  background:radial-gradient(120% 100% at 50% 50%, transparent 55%, rgba(0,0,0,.55) 100%);
}
.main .block-container{ padding-top:1.2rem; padding-bottom:2rem; max-width:1440px; }
header[data-testid="stHeader"]{ background:transparent; }
footer,#MainMenu{ visibility:hidden; }

/* ── Hero terminal ── */
.ah-hero{
  display:flex; align-items:center; gap:18px;
  border:1px solid var(--line); border-radius:4px;
  padding:16px 20px; margin-bottom:18px;
  background:linear-gradient(180deg, rgba(255,176,0,.05), transparent);
  box-shadow:inset 0 0 40px rgba(255,176,0,.04);
}
.ah-hero img{ width:50px; height:50px; object-fit:contain; filter:drop-shadow(0 0 6px rgba(255,176,0,.5)); }
.ah-hero .t{
  font-family:'Chakra Petch',sans-serif; font-weight:700; font-size:1.5rem;
  letter-spacing:.06em; text-shadow:var(--glow); margin:0; line-height:1;
}
.ah-hero .s{ color:var(--muted); font-size:.72rem; letter-spacing:.18em; margin-top:5px; text-transform:uppercase; }
.ah-hero .stat{ margin-left:auto; text-align:right; }
.ah-hero .stat .dot{ color:var(--amber); animation:blink 1.4s steps(1) infinite; }
.ah-hero .stat .lbl{ color:var(--muted); font-size:.65rem; letter-spacing:.14em; }
@keyframes blink{ 50%{ opacity:0 } }

/* ── Section labels (terminal) ── */
.ah-sec{
  font-family:'Chakra Petch',sans-serif; font-weight:600; font-size:.78rem;
  letter-spacing:.22em; text-transform:uppercase; color:var(--amber);
  margin:6px 0 12px; display:flex; align-items:center; gap:8px;
}
.ah-sec::before{ content:"//"; color:var(--muted); }
.ah-sec .hint{ color:var(--muted); letter-spacing:.08em; font-weight:400; margin-left:auto; font-size:.7rem;}

/* ── KPI readouts ── */
.ah-kpis{ display:grid; grid-template-columns:repeat(4,1fr); gap:12px; margin:6px 0 20px; }
.ah-kpi{
  border:1px solid var(--line); border-radius:4px; padding:14px 16px;
  background:linear-gradient(180deg, rgba(255,176,0,.04), transparent);
  position:relative;
}
.ah-kpi::before{
  content:""; position:absolute; top:10px; right:12px; width:6px; height:6px;
  border-radius:50%; background:var(--amber); box-shadow:var(--glow);
}
.ah-kpi .l{ color:var(--muted); font-size:.62rem; letter-spacing:.16em; text-transform:uppercase; }
.ah-kpi .v{
  font-family:'Chakra Petch',sans-serif; font-weight:700; font-size:2rem;
  margin-top:4px; text-shadow:var(--glow); line-height:1;
}
.ah-kpi .h{ color:var(--muted); font-size:.66rem; margin-top:5px; letter-spacing:.04em; }

/* ── Radar AHORA ── */
.ah-now{
  display:grid; grid-template-columns:340px 1fr; gap:26px; align-items:center;
  border:1px solid var(--line); border-radius:6px; padding:24px;
  background:
    radial-gradient(80% 80% at 22% 50%, rgba(255,176,0,.06), transparent 60%),
    var(--panel);
  margin-bottom:22px;
}
.radar{
  position:relative; width:300px; height:300px; margin:0 auto;
  border-radius:50%; border:1px solid rgba(255,176,0,.35);
  box-shadow:inset 0 0 40px rgba(255,176,0,.08), 0 0 30px rgba(255,176,0,.06);
}
.radar .ring{ position:absolute; border:1px solid rgba(255,176,0,.16); border-radius:50%; top:50%; left:50%; transform:translate(-50%,-50%); }
.radar .r1{ width:66%; height:66%; } .radar .r2{ width:33%; height:33%; }
.radar .cross{ position:absolute; background:rgba(255,176,0,.14); }
.radar .cx{ top:50%; left:0; width:100%; height:1px; transform:translateY(-.5px); }
.radar .cy{ left:50%; top:0; height:100%; width:1px; transform:translateX(-.5px); }
.radar .sweep{
  position:absolute; inset:0; border-radius:50%;
  background:conic-gradient(from 0deg, rgba(255,176,0,.45), rgba(255,176,0,.02) 50deg, transparent 90deg);
  animation:sweep 3.2s linear infinite; mix-blend-mode:screen;
}
@keyframes sweep{ to{ transform:rotate(360deg) } }
.blip{
  position:absolute; transform:translate(-50%,-50%); border-radius:50%;
  box-shadow:0 0 10px currentColor; animation:pulse 2s ease-in-out infinite;
}
.blip .bl{
  position:absolute; left:50%; top:120%; transform:translateX(-50%);
  font-size:.55rem; letter-spacing:.1em; text-transform:uppercase; color:var(--muted); white-space:nowrap;
}
@keyframes pulse{ 50%{ opacity:.45 } }

.ah-now .read .status{ color:var(--muted); font-size:.7rem; letter-spacing:.16em; }
.ah-now .read .status b{ color:var(--amber); }
.ah-now .read .big{
  font-family:'Chakra Petch',sans-serif; font-weight:700;
  font-size:4.4rem; line-height:.95; text-shadow:var(--glow); margin:6px 0 2px;
}
.ah-now .read .zone{
  font-family:'Chakra Petch',sans-serif; font-size:1.6rem; font-weight:600;
  text-transform:uppercase; letter-spacing:.08em;
}
.ah-now .read .meta{ color:var(--muted); font-size:.74rem; margin-top:10px; letter-spacing:.05em; }
.ah-now .read .peak{
  margin-top:14px; padding:9px 13px; border-left:2px solid var(--amber);
  background:rgba(255,176,0,.05); font-size:.78rem; letter-spacing:.04em;
}
.ah-now .read .peak b{ color:var(--amber); }

/* ── Zone ranking bars ── */
.ah-zrow{
  display:flex; align-items:center; gap:12px; padding:9px 12px; margin-bottom:5px;
  border:1px solid var(--line); border-radius:4px; background:rgba(255,176,0,.03);
}
.ah-zrow .n{ flex:0 0 96px; text-transform:uppercase; font-size:.78rem; letter-spacing:.06em; }
.ah-zrow .b{ flex:1; height:7px; background:rgba(255,176,0,.08); border-radius:0; overflow:hidden; }
.ah-zrow .bf{ height:100%; }
.ah-zrow .p{ flex:0 0 56px; text-align:right; font-weight:700; font-size:.86rem; }

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"]{ gap:2px; border-bottom:1px solid var(--line); }
.stTabs [data-baseweb="tab"]{
  background:transparent; padding:9px 18px; color:var(--muted);
  font-family:'Chakra Petch',sans-serif; font-weight:600; font-size:.8rem;
  letter-spacing:.12em; text-transform:uppercase; border-radius:0;
}
.stTabs [aria-selected="true"]{ color:var(--amber); border-bottom:2px solid var(--amber) !important; text-shadow:var(--glow); }

/* ── Sidebar ── */
[data-testid="stSidebar"]{ background:var(--bg2); border-right:1px solid var(--line); }
[data-testid="stSidebar"] img{ filter:drop-shadow(0 0 6px rgba(255,176,0,.4)); }
.ah-side{ font-family:'Chakra Petch',sans-serif; font-size:.66rem; font-weight:700;
  letter-spacing:.2em; text-transform:uppercase; color:var(--muted); margin:16px 0 6px; }
[data-testid="stSidebar"] hr{ border-color:var(--line); }

/* Inputs */
.stDateInput input,.stTextInput input{ background:var(--panel)!important; color:var(--text)!important; border:1px solid var(--line)!important; border-radius:2px!important; }
.stButton button{
  background:transparent; color:var(--amber)!important; border:1px solid var(--amber)!important;
  border-radius:2px; font-family:'Chakra Petch',sans-serif; font-weight:600;
  letter-spacing:.1em; text-transform:uppercase; transition:all .15s;
}
.stButton button:hover{ background:var(--amber); color:#0A0A0B!important; box-shadow:var(--glow); }

[data-testid="stChatMessage"]{ background:var(--panel); border:1px solid var(--line); border-radius:4px; }
[data-testid="stDataFrame"]{ border:1px solid var(--line); }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
# Data
# ════════════════════════════════════════════════════════════════════════════
@st.cache_data(show_spinner=False)
def load_events() -> pd.DataFrame:
    if not EVENTS_PATH.exists():
        return pd.DataFrame()
    df = pd.read_csv(EVENTS_PATH)
    df["fecha_evento"] = pd.to_datetime(df["fecha"].fillna(df["pub_date"]), errors="coerce", utc=True).dt.tz_localize(None)
    return df


@st.cache_resource(show_spinner=False)
def load_model():
    if not MODEL_PATH.exists():
        return None
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


@st.cache_resource(show_spinner=False)
def load_engine():
    """Motor RAG (hybrid + routing + rerank). Cacheado entre reruns."""
    try:
        sys.path.insert(0, str(ROOT / "app"))
        from rag_engine import RAGEngine
        return RAGEngine()
    except Exception as e:
        st.warning(f"RAG no disponible: {e}")
        return None


def zona_prefix(model_bundle) -> str:
    """Detecta si el modelo usa 'zona12_' (12 zonas) o 'zona_' (6 zonas)."""
    feats = model_bundle["features"]
    return "zona12_" if any(c.startswith("zona12_") for c in feats) else "zona_"


def active_coords(model_bundle) -> dict:
    """Detecta el esquema (6 ó 12 zonas) por pertenencia de los nombres de zona del modelo."""
    pref = zona_prefix(model_bundle)
    zones = [c[len(pref):] for c in model_bundle["features"] if c.startswith(pref)]
    n12 = sum(z in ZONA12_COORDS for z in zones)
    n6 = sum(z in ZONA_COORDS for z in zones)
    return ZONA12_COORDS if n12 >= n6 else ZONA_COORDS


def predict_zone_probabilities(model_bundle, target: date, slot: int) -> dict[str, float]:
    if model_bundle is None:
        return {}
    from workalendar.europe.spain import CanaryIslands
    cal = CanaryIslands()
    es_festivo = int(not cal.is_working_day(target))
    vispera    = int(not cal.is_working_day(target + timedelta(days=1)))
    dow = target.weekday()
    feats = model_bundle["features"]
    pref = zona_prefix(model_bundle)
    zones = [c[len(pref):] for c in feats if c.startswith(pref)]
    rows = []
    for z in zones:
        row = {"dia_semana": dow, "mes": target.month, "anio": target.year, "slot": slot,
               "es_finde": int(dow >= 5), "es_festivo": es_festivo, "vispera_festivo": vispera}
        for zc in zones:
            row[f"{pref}{zc}"] = 1 if zc == z else 0
        rows.append(row)
    X = pd.DataFrame(rows).reindex(columns=feats, fill_value=0)
    probs = model_bundle["model"].predict_proba(X)[:, 1]
    return dict(zip(zones, probs))


@st.cache_data(show_spinner=False)
def peak_insight(df: pd.DataFrame) -> str:
    if df.empty:
        return ""
    c = df[df["es_control"] == True].copy()
    if c.empty:
        return ""
    c["dow"] = c["fecha_evento"].dt.dayofweek
    c["h"] = c["fecha_evento"].dt.hour
    dias = ["LUNES", "MARTES", "MIÉRCOLES", "JUEVES", "VIERNES", "SÁBADO", "DOMINGO"]
    top_dow = int(c["dow"].mode().iloc[0])
    top_h = int(c["h"].mode().iloc[0])
    top_zone = str(c["zona"].mode().iloc[0]).upper() if "zona" in c else "?"
    return f"{dias[top_dow]} {top_h:02d}:00 · {top_zone}"


def color_for(p: float, vmax: float) -> str:
    r = p / vmax if vmax else 0
    if r > 0.66: return DANGER
    if r > 0.33: return AMBER
    return AMBER_2


def logo_b64():
    return base64.b64encode(LOGO_PATH.read_bytes()).decode() if LOGO_PATH.exists() else None


def render_radar(probs: dict[str, float], coords: dict) -> str:
    """Radar con blips posicionados según la geografía real de cada zona."""
    lat0, lon0 = LANZAROTE_CENTER
    vmax = max(probs.values()) if probs else 1
    blips = []
    probs = {z: p for z, p in probs.items() if z in coords}
    # distancia geográfica máx para normalizar
    dists = {z: math.hypot(coords[z][0]-lat0, coords[z][1]-lon0) for z in probs}
    dmax = max(dists.values()) if dists else 1
    for z, p in probs.items():
        lat, lon = coords[z]
        ang = math.atan2(lon - lon0, lat - lat0)          # 0 = norte, horario
        r = 12 + (dists[z] / dmax) * 32                    # % radio (12..44)
        top = 50 - math.cos(ang) * r
        left = 50 + math.sin(ang) * r
        size = 7 + (p / vmax) * 14
        col = color_for(p, vmax)
        blips.append(
            f'<div class="blip" style="top:{top:.1f}%;left:{left:.1f}%;'
            f'width:{size:.0f}px;height:{size:.0f}px;background:{col};color:{col}">'
            f'<span class="bl">{z.replace("_", " ")}</span></div>'
        )
    return (
        '<div class="radar"><div class="ring r1"></div><div class="ring r2"></div>'
        '<div class="cross cx"></div><div class="cross cy"></div>'
        '<div class="sweep"></div>' + "".join(blips) + '</div>'
    )


# ════════════════════════════════════════════════════════════════════════════
# Sidebar
# ════════════════════════════════════════════════════════════════════════════
if LOGO_PATH.exists():
    st.sidebar.image(str(LOGO_PATH), use_container_width=True)

st.sidebar.markdown('<div class="ah-side">Objetivo temporal</div>', unsafe_allow_html=True)
selected_date = st.sidebar.date_input("Día", value=date.today(), label_visibility="collapsed")
st.sidebar.markdown('<div class="ah-side">Hora</div>', unsafe_allow_html=True)
selected_hour = st.sidebar.slider("Hora", 0, 23, 22, label_visibility="collapsed")
selected_slot = selected_hour // SLOT_HOURS
slot_range = f"{selected_slot*SLOT_HOURS:02d}:00—{(selected_slot+1)*SLOT_HOURS:02d}:00"
st.sidebar.markdown(
    f'<div style="color:#9A865C;font-size:.74rem;margin-top:4px">SLOT &gt; <b style="color:#FFB000">{slot_range}</b></div>',
    unsafe_allow_html=True)

st.sidebar.markdown("---")
st.sidebar.markdown('<div class="ah-side">Comandos</div>', unsafe_allow_html=True)
if st.sidebar.button("⟳ Sincronizar Telegram", use_container_width=True):
    with st.spinner("Interceptando nuevos reportes…"):
        r = subprocess.run([sys.executable, str(SCRAPER), "--no-auth"],
                           capture_output=True, text=True, cwd=str(ROOT))
        st.sidebar.text(r.stdout[-500:])
        if r.returncode != 0:
            st.sidebar.error(r.stderr[-300:])

st.sidebar.markdown("---")
st.sidebar.markdown(
    '<div style="color:#9A865C;font-size:.7rem;line-height:1.6;padding:6px 0">'
    '<b style="color:#FFD9A0">[ AVISO ]</b><br>Análisis de patrones sobre datos públicos. '
    'Modelo histórico 2015-2021. Uso educativo. No usar para evadir controles.</div>',
    unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
# Hero
# ════════════════════════════════════════════════════════════════════════════
df = load_events()
model_bundle = load_model()
metrics_json = json.loads(METRICS_PATH.read_text()) if METRICS_PATH.exists() else {}

lb = logo_b64()
logo_img = f'<img src="data:image/png;base64,{lb}"/>' if lb else ""
st.markdown(f"""
<div class="ah-hero">
  {logo_img}
  <div>
    <div class="t">ANTI HOG <span style="color:#9A865C">// INTERCEPT</span></div>
    <div class="s">Sistema predictivo de controles · Lanzarote</div>
  </div>
  <div class="stat">
    <div class="lbl"><span class="dot">●</span> ONLINE</div>
    <div style="font-family:'Chakra Petch';color:#FFB000;font-size:.9rem;letter-spacing:.1em">{datetime.now():%H:%M:%S}</div>
  </div>
</div>
""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
# Panel AHORA (radar en vivo)
# ════════════════════════════════════════════════════════════════════════════
now = datetime.now()
now_slot = now.hour // SLOT_HOURS
now_probs = predict_zone_probabilities(model_bundle, now.date(), now_slot)

if now_probs:
    coords_now = active_coords(model_bundle)
    top_zone, top_p = max(now_probs.items(), key=lambda x: x[1])
    vmax = max(now_probs.values())
    radar_html = render_radar(now_probs, coords_now)
    top_zone = top_zone.replace("_", " ")
    peak = peak_insight(df)
    now_range = f"{now_slot*SLOT_HOURS:02d}:00—{(now_slot+1)*SLOT_HOURS:02d}:00"
    rel = top_p / vmax if vmax else 0
    level = "ALTO" if rel > 0.66 else "MEDIO" if rel > 0.33 else "BAJO"
    level_col = DANGER if rel > 0.66 else AMBER if rel > 0.33 else AMBER_2

    st.markdown(f"""
    <div class="ah-now">
      <div>{radar_html}</div>
      <div class="read">
        <div class="status"><b>● SCANNING</b> &nbsp;·&nbsp; {now:%A %d/%m} &nbsp;·&nbsp; {now_range}</div>
        <div class="big" style="color:{level_col}">{top_p*100:.0f}<span style="font-size:2rem">%</span></div>
        <div class="zone">{top_zone} <span style="color:{level_col};font-size:1rem">[{level}]</span></div>
        <div class="meta">Zona con mayor probabilidad de control en la franja horaria actual</div>
        <div class="peak">PICO HISTÓRICO &gt; <b>{peak}</b></div>
      </div>
    </div>
    """, unsafe_allow_html=True)
else:
    st.info("Sube el modelo (`models/predictor.pkl`) para activar el radar en vivo.")


# ════════════════════════════════════════════════════════════════════════════
# KPIs
# ════════════════════════════════════════════════════════════════════════════
total_msgs = len(df)
total_controls = int((df["es_control"] == True).sum()) if not df.empty else 0
total_days = df["fecha_evento"].dt.date.nunique() if not df.empty else 0
auc = metrics_json.get("roc_auc_test") or metrics_json.get("roc_auc")
auc_str = f"{auc:.3f}" if auc else "—"

st.markdown(f"""
<div class="ah-kpis">
  <div class="ah-kpi"><div class="l">Reportes analizados</div><div class="v">{total_msgs:,}</div><div class="h">interceptados desde 2015</div></div>
  <div class="ah-kpi"><div class="l">Controles detectados</div><div class="v">{total_controls:,}</div><div class="h">extraídos con LLM</div></div>
  <div class="ah-kpi"><div class="l">Días en registro</div><div class="v">{total_days:,}</div><div class="h">con actividad</div></div>
  <div class="ah-kpi"><div class="l">Precisión modelo</div><div class="v">{auc_str}</div><div class="h">ROC-AUC · XGBoost</div></div>
</div>
""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
# Tabs
# ════════════════════════════════════════════════════════════════════════════
PLOTLY = dict(
    template="plotly_dark", paper_bgcolor="#15130E", plot_bgcolor="#15130E",
    font=dict(family="JetBrains Mono, monospace", color=SAFE),
    margin=dict(l=20, r=20, t=50, b=20),
    colorway=[AMBER, AMBER_2, DANGER, "#C98A00", "#FFD9A0", "#8A865C"],
)

tab_mapa, tab_chat, tab_stats = st.tabs(["Mapa táctico", "Asistente", "Inteligencia"])

with tab_mapa:
    top_l, top_r = st.columns([3, 1.4])
    with top_l:
        st.markdown(
            f'<div class="ah-sec">Probabilidad por zona <span class="hint">{selected_date} · {slot_range}</span></div>',
            unsafe_allow_html=True)
    with top_r:
        vista = st.radio("Vista", ["Zonas", "Mapa de calor"], horizontal=True,
                         label_visibility="collapsed")

    if model_bundle is None:
        st.info("Modelo no cargado. Súbelo a `models/predictor.pkl`.")
    else:
        coords = active_coords(model_bundle)
        probs = predict_zone_probabilities(model_bundle, selected_date, selected_slot)
        probs = {z: p for z, p in probs.items() if z in coords}
        sorted_z = sorted(probs.items(), key=lambda x: -x[1])
        vmax = max(probs.values()) if probs else 1

        col_map, col_list = st.columns([2.2, 1])
        with col_map:
            m = folium.Map(location=LANZAROTE_CENTER, zoom_start=11,
                           tiles="https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png",
                           attr="© OpenStreetMap, © CARTO")
            if vista == "Zonas":
                for z, p in sorted_z:
                    lat, lon = coords.get(z, LANZAROTE_CENTER)
                    col = color_for(p, vmax)
                    folium.CircleMarker((lat, lon), radius=14 + (p/vmax)*34,
                        popup=folium.Popup(f"<b>{z.replace('_',' ').upper()}</b><br>P={p:.2%}", max_width=170),
                        tooltip=f"{z.replace('_',' ')}: {p:.1%}", color=col, weight=2,
                        fill=True, fill_color=col, fill_opacity=0.5).add_to(m)
            else:  # Mapa de calor con los puntos reales geocodificados
                from folium.plugins import HeatMap
                pts = df[(df["es_control"] == True) & df["lat"].notna() & df["lon"].notna()]
                heat = pts[["lat", "lon"]].values.tolist()
                if heat:
                    HeatMap(heat, radius=18, blur=22, min_opacity=0.3,
                            gradient={0.2: "#FF7A00", 0.5: "#FFB000", 0.9: "#FF3B30"}).add_to(m)
            st_folium(m, height=560, returned_objects=[], use_container_width=True)

        with col_list:
            if vista == "Zonas":
                st.markdown('<div class="ah-sec">Ranking de riesgo</div>', unsafe_allow_html=True)
                for z, p in sorted_z:
                    col = color_for(p, vmax)
                    w = (p/vmax)*100
                    st.markdown(
                        f'<div class="ah-zrow"><span class="n">{z.replace("_"," ")}</span>'
                        f'<span class="b"><span class="bf" style="width:{w:.0f}%;background:{col};box-shadow:0 0 8px {col}"></span></span>'
                        f'<span class="p" style="color:{col}">{p*100:.1f}%</span></div>',
                        unsafe_allow_html=True)
            else:
                st.markdown('<div class="ah-sec">Puntos calientes</div>', unsafe_allow_html=True)
                hot = (df[df["es_control"] == True]["canonical"].value_counts().head(12))
                hmax = int(hot.max()) if len(hot) else 1
                for name, cnt in hot.items():
                    w = (cnt / hmax) * 100
                    st.markdown(
                        f'<div class="ah-zrow"><span class="n" style="flex:0 0 130px">{str(name).replace("_"," ")}</span>'
                        f'<span class="b"><span class="bf" style="width:{w:.0f}%;background:{AMBER};box-shadow:0 0 8px {AMBER}"></span></span>'
                        f'<span class="p" style="color:{AMBER}">{int(cnt)}</span></div>',
                        unsafe_allow_html=True)

with tab_chat:
    st.markdown('<div class="ah-sec">Asistente histórico <span class="hint">hybrid · routing · re-rank</span></div>',
                unsafe_allow_html=True)
    engine = load_engine()
    if engine is None:
        st.info("RAG no construido. Corre `python scripts/05_build_rag.py`.")
    else:
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []
        cols = st.columns(3)
        examples = [
            "¿Controles en Arrecife los viernes noche?",
            "¿Cuántos controles de alcoholemia en Playa Blanca?",
            "¿Qué día de la semana hay más controles?",
        ]
        for i, ex in enumerate(examples):
            if cols[i].button(ex, use_container_width=True, key=f"ex_{i}"):
                st.session_state._pending = ex
        for q, a in st.session_state.chat_history:
            with st.chat_message("user"): st.write(q)
            with st.chat_message("assistant"): st.markdown(a, unsafe_allow_html=True)
        question = st.chat_input("Consulta el histórico…")
        if st.session_state.get("_pending"):
            question = st.session_state.pop("_pending")
        if question:
            with st.chat_message("user"): st.write(question)
            with st.chat_message("assistant"):
                with st.spinner("Rastreando histórico…"):
                    from openai import OpenAI
                    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
                    result = engine.answer(question, openai_client=client,
                                           model=os.environ.get("OPENAI_MODEL", "gpt-4o-mini"))
                    # badges de routing/filtros
                    f = result.filters
                    chips = [f'ruta: <b>{result.route}</b>']
                    if f.zona: chips.append(f'zona: {f.zona}')
                    if f.tipo: chips.append(f'tipo: {f.tipo}')
                    if f.anio: chips.append(f'año: {f.anio}')
                    st.markdown(
                        '<div style="font-size:.66rem;letter-spacing:.08em;color:#9A865C;margin-bottom:6px">'
                        + '  ·  '.join(chips) + '</div>', unsafe_allow_html=True)
                    st.write(result.answer)
                    if result.sources:
                        with st.expander(f"Mensajes recuperados ({len(result.sources)})"):
                            for s in result.sources:
                                st.markdown(f"`{s.get('date','?')}` **{s.get('zona','?')}** · {s['text']}")
                st.session_state.chat_history.append((question, result.answer))

with tab_stats:
    st.markdown('<div class="ah-sec">Inteligencia del histórico</div>', unsafe_allow_html=True)
    if df.empty:
        st.info("Sin datos.")
    else:
        c = df[df["es_control"] == True].copy()
        if c.empty:
            st.warning("Sin eventos de control.")
        else:
            D = {"Monday":"L","Tuesday":"M","Wednesday":"X","Thursday":"J","Friday":"V","Saturday":"S","Sunday":"D"}
            c["dow"] = c["fecha_evento"].dt.day_name().map(D)
            c["mes"] = c["fecha_evento"].dt.month
            c["h"] = c["fecha_evento"].dt.hour
            c1, c2 = st.columns(2)
            with c1:
                f = px.histogram(c, x="dow", category_orders={"dow": ["L","M","X","J","V","S","D"]},
                                 title="CONTROLES POR DÍA")
                f.update_layout(**PLOTLY); f.update_traces(marker_color=AMBER); st.plotly_chart(f, use_container_width=True)
            with c2:
                f = px.histogram(c, x="h", nbins=24, title="DISTRIBUCIÓN HORARIA")
                f.update_layout(**PLOTLY, xaxis_title="hora"); f.update_traces(marker_color=AMBER); st.plotly_chart(f, use_container_width=True)
            c3, c4 = st.columns(2)
            with c3:
                f = px.histogram(c, x="mes", nbins=12, title="CONTROLES POR MES")
                f.update_layout(**PLOTLY); f.update_traces(marker_color=AMBER); st.plotly_chart(f, use_container_width=True)
            with c4:
                if "zona" in c.columns:
                    f = px.histogram(c, x="zona", color="tipo_control", title="POR ZONA Y TIPO")
                    f.update_layout(**PLOTLY); st.plotly_chart(f, use_container_width=True)
