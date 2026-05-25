"""
Construye el índice RAG v2 con embeddings enriquecidos + modelo e5.

Mejoras vs v1:
- Modelo: intfloat/multilingual-e5-base (mejor recall multilingüe que MiniLM).
- Texto embebido enriquecido con contexto: "passage: [fecha | zona | tipo] mensaje"
  (el mensaje crudo es muy corto ~44 chars; el contexto mejora la recuperación).
- Metadatos para filtrado híbrido: date, zona, tipo_control, lugar, lat, lon, es_control.

Colección: antihog_v2  (la app usa v2 si existe, si no cae a v1).

Uso:
    python scripts/05_build_rag.py [--batch 128]
"""

import argparse
import json
from pathlib import Path

import pandas as pd
from chromadb import PersistentClient
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
MESSAGES_PATH = ROOT / "data" / "raw" / "messages.json"
EVENTS_PATH = ROOT / "data" / "processed" / "events_geocoded.csv"
CHROMA_DIR = ROOT / "rag" / "chroma_db"
CHROMA_DIR.mkdir(parents=True, exist_ok=True)

COLLECTION = "antihog_v2"
EMBED_MODEL = "intfloat/multilingual-e5-base"


def load_messages() -> pd.DataFrame:
    raw = pd.DataFrame(json.loads(MESSAGES_PATH.read_text()))
    if EVENTS_PATH.exists():
        events = pd.read_csv(EVENTS_PATH)
        meta_cols = ["message_id", "es_control", "fecha", "hora", "lugar",
                     "tipo_control", "canonical", "zona", "lat", "lon"]
        keep = [c for c in meta_cols if c in events.columns]
        events = events[keep].rename(columns={"message_id": "id"})
        raw = raw.merge(events, on="id", how="left")
    return raw


def enrich(row) -> str:
    """Texto contextualizado para embeber (mejora la recuperación de mensajes cortos)."""
    date = str(row.get("date", ""))[:10]
    zona = row.get("zona") if pd.notna(row.get("zona")) else ""
    tipo = row.get("tipo_control") if pd.notna(row.get("tipo_control")) else ""
    ctx = " | ".join(x for x in [date, zona, tipo] if x)
    text = str(row["text"])
    return f"[{ctx}] {text}" if ctx else text


def main(batch: int) -> None:
    df = load_messages()
    df = df[df["text"].notna() & (df["text"].str.len() > 0)]
    print(f"[rag] mensajes a indexar: {len(df)}")

    print(f"[rag] cargando modelo: {EMBED_MODEL} (puede descargar ~450MB la 1ª vez)")
    model = SentenceTransformer(EMBED_MODEL)

    client = PersistentClient(path=str(CHROMA_DIR))
    try:
        client.delete_collection(COLLECTION)
    except Exception:
        pass
    coll = client.create_collection(COLLECTION, metadata={"hnsw:space": "cosine"})

    ids, docs, metas, embed_texts = [], [], [], []
    for _, row in df.iterrows():
        meta = {"date": str(row.get("date", ""))[:10] or ""}
        for k in ("es_control", "zona", "canonical", "tipo_control", "lugar", "lat", "lon"):
            v = row.get(k)
            if pd.notna(v) and v != "":
                if k in ("lat", "lon"):
                    meta[k] = float(v)
                elif k == "es_control":
                    meta[k] = bool(v)
                else:
                    meta[k] = str(v)
        ids.append(str(row["id"]))
        docs.append(str(row["text"]))            # documento mostrado = texto original
        embed_texts.append("passage: " + enrich(row))  # texto embebido = enriquecido (prefijo e5)
        metas.append(meta)

    print(f"[rag] embebiendo en lotes de {batch}…")
    for i in tqdm(range(0, len(embed_texts), batch), desc="embed"):
        et = embed_texts[i:i + batch]
        embs = model.encode(et, show_progress_bar=False, normalize_embeddings=True).tolist()
        coll.add(ids=ids[i:i+batch], documents=docs[i:i+batch],
                 embeddings=embs, metadatas=metas[i:i+batch])

    print(f"[rag] colección '{COLLECTION}' = {coll.count()} vectores")
    print(f"[rag] listo en {CHROMA_DIR}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=128)
    args = parser.parse_args()
    main(args.batch)
