<div align="center">
  <img src="assets/logo.png" alt="Anti Hog" width="180"/>

  # Anti Hog

  **Predicción de controles policiales en Lanzarote mediante análisis de patrones espacio-temporales sobre datos públicos de Telegram.**

  `NLP` · `Machine Learning` · `RAG` · `Geovisualización`
</div>

---

> [!NOTE]
> **Enmarcado académico.** Proyecto del Grado Superior de Inteligencia Artificial. Es un ejercicio de IA aplicada de principio a fin: extracción de entidades con LLM, predicción con ML y recuperación de información. **No es una herramienta de evasión** — analiza patrones históricos sobre reportes públicos de usuarios, con fines educativos.

## ¿Qué hace?

Anti Hog coge un canal público de Telegram donde los vecinos avisan de controles (mensajes cortos, en lenguaje de calle, sin estructura) y construye un sistema completo que:

- **Descarga y mantiene el histórico** del canal de forma automática y desatendida.
- **Estructura el texto libre** con un LLM, sacando fecha, hora, lugar y tipo de control de cada mensaje.
- **Geolocaliza** los lugares informales ("rotonda del Gallo Loco") sobre el mapa de Lanzarote.
- **Aprende los patrones** y predice la probabilidad de control por zona, día y franja horaria.
- **Responde preguntas en lenguaje natural** sobre el histórico mediante un RAG.
- Lo presenta todo en una **app con estética de radar** y mapa interactivo.

## Funcionalidades

### Scraping autónomo
- Descarga del histórico completo del canal vía **Telethon** (API oficial de Telegram).
- Resolución del canal por su nombre y **fusión incremental sin duplicados** (snapshots con copia de seguridad de cada ejecución).
- **Tarea programada cada 20 h** (`launchd`): el dataset crece solo mientras el canal siga activo.

### Extracción de entidades con LLM
- Cada mensaje pasa por **gpt-4o-mini** y devuelve un JSON estructurado: `{es_control, fecha, hora, lugar, tipo_control, confianza}`.
- Ejecución **asíncrona y concurrente** con **caché por hash** del texto: no se repite ni se paga dos veces la misma llamada.
- De ~16.000 mensajes se identifican **8.444 controles reales**.

### Geocoding
- **Gazetteer manual de Lanzarote** (+75 lugares: rotondas, pueblos, núcleos) con **fuzzy matching** (rapidfuzz) para los mal escritos.
- **Doble esquema de zonas**: 6 zonas amplias (documento/estadística) y **12 zonas finas** (modelo). ~72% de eventos geolocalizados.

### Modelo predictivo
- Problema modelado por **celdas** (`fecha × zona × franja de 3 h`) → clasificación binaria con **XGBoost**.
- **Validación temporal** (entrena con el pasado, valida con el futuro) — sin fuga de información, a diferencia de un k-fold normal.
- **Optuna** para hiperparámetros + **early stopping** + optimización del umbral.
- **Detección automática del gap** de datos (el canal estuvo inactivo ~5 años): descarta el periodo muerto solo.
- Resultado: **ROC-AUC 0.79** sobre 12 zonas.

### RAG (asistente sobre el histórico)
- Embeddings con **e5-base** y *embeddings enriquecidos* (fecha · zona · tipo + texto), almacenados en **ChromaDB**.
- **Filtrado híbrido**: detecta si la pregunta es por una zona/tipo/año y filtra por metadatos antes de buscar.
- **Router estadístico**: las preguntas de "cuántos…" se resuelven con un cálculo exacto en pandas, no con el LLM.
- **Re-ranking** con cross-encoder + deduplicación.
- Evaluación reproducible (recall@8 + MRR): **MRR 0.78 → 1.0** tras las mejoras.

### App (Streamlit + Folium)
- Estética **radar de fósforo ámbar** (CRT, scanlines, tipografías Chakra Petch + JetBrains Mono).
- **Panel "AHORA"**: radar en vivo con el riesgo de la franja actual y un *blip* por zona según su posición geográfica real.
- **Mapa táctico**: vista por zonas o mapa de calor de puntos reales + ranking de puntos calientes.
- **Asistente** RAG en lenguaje natural y pestaña de **estadísticas**.

## Pipeline

```
Telegram ──► scraping (Telethon) ──────────► mensajes JSON
         ──► extracción LLM (gpt-4o-mini) ──► eventos estructurados (CSV)
         ──► geocoding (gazetteer + fuzzy) ─► eventos con coordenadas
         ──► XGBoost (validación temporal) ─► modelo predictivo
         ──► embeddings e5 + ChromaDB ──────► RAG para Q&A
         ──► Streamlit + Folium ────────────► app con radar y mapa
```

## Estructura

```
.
├── scripts/
│   ├── 01_scrape_telegram.py    # Telethon: histórico + incremental
│   ├── 02_extract_entities.py   # LLM async → events.csv
│   ├── 03_geocode.py            # gazetteer Lanzarote + fuzzy → coords
│   ├── 04_train_model.py        # XGBoost + validación temporal
│   ├── 05_build_rag.py          # embeddings e5 + ChromaDB
│   └── eval_rag.py              # recall@8 + MRR
├── notebooks/
│   └── train_colab.ipynb        # entrenamiento en Colab (Optuna, early stopping)
├── app/
│   ├── streamlit_app.py         # entrypoint (radar + mapa + chat + stats)
│   └── rag_engine.py            # motor RAG (filtrado, routing, rerank)
├── models/                      # metrics.json, train_report.png (predictor.pkl no versionado)
├── presentacion/                # guion de defensa
├── data/                        # generado por el pipeline (no versionado)
└── requirements.txt
```

> Los datos en crudo, la caché, el modelo entrenado y el vector store **no se versionan** (privacidad de terceros + se regeneran ejecutando el pipeline).

## Puesta en marcha

```bash
# 1. Entorno
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. Credenciales
cp .env.example .env
# editar .env con tus claves de Telegram (my.telegram.org) y OpenAI
```

### Ejecutar el pipeline

```bash
python scripts/01_scrape_telegram.py     # descarga el histórico
python scripts/02_extract_entities.py    # LLM → events.csv
python scripts/03_geocode.py             # geocoding → events_geocoded.csv
python scripts/04_train_model.py         # entrena el modelo
python scripts/05_build_rag.py           # construye el RAG

streamlit run app/streamlit_app.py       # lanza la app
```

> El modelo también puede entrenarse en Google Colab con `notebooks/train_colab.ipynb` (más rápido).

## Stack

`Python 3.12` · `Telethon` · `OpenAI gpt-4o-mini` · `pandas` · `XGBoost` · `Optuna` · `workalendar` · `rapidfuzz` · `sentence-transformers (e5)` · `ChromaDB` · `Streamlit` · `Folium`

## Limitaciones

- El histórico incluye los años de COVID, con patrones no del todo extrapolables a hoy.
- Datos sesgados por quién reporta: es lo que la gente avisó, no un censo objetivo.
- Las probabilidades sirven para **ordenar zonas por riesgo**, no como porcentaje calibrado.
