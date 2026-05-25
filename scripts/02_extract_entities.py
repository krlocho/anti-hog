"""
Extrae con gpt-4o-mini, para cada mensaje de data/raw/messages.json:
  {es_control, fecha, hora, lugar, tipo_control, confianza}

Caching por hash del texto en data/processed/extraction_cache.json (idempotente).
Ejecuta en concurrencia con asyncio (CONCURRENCY workers) para procesar
15k+ mensajes en minutos en lugar de horas.

Salida: data/processed/events.csv

Uso:
    python scripts/02_extract_entities.py [--limit N] [--concurrency C]
"""

import argparse
import asyncio
import hashlib
import json
import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from openai import AsyncOpenAI
from tqdm.asyncio import tqdm_asyncio

load_dotenv()
client = AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"])
MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")

ROOT = Path(__file__).resolve().parents[1]
IN_PATH = ROOT / "data" / "raw" / "messages.json"
OUT_PATH = ROOT / "data" / "processed" / "events.csv"
CACHE_PATH = ROOT / "data" / "processed" / "extraction_cache.json"
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

SYSTEM_PROMPT = """Eres un extractor de eventos de mensajes en español sobre controles policiales en Lanzarote (Canarias).
Dado un mensaje y su fecha de publicación, extrae UN único evento principal con esta estructura JSON:

{
  "es_control": true|false,
  "fecha": "YYYY-MM-DD" | null,
  "hora": "HH:MM" | null,
  "lugar": string | null,
  "tipo_control": "alcoholemia"|"trafico"|"drogas"|"documentacion"|"radar"|"otro"|null,
  "confianza": 0.0-1.0
}

Reglas:
- es_control=false si el mensaje es saludo, agradecimiento, broma, anuncio, off-topic o metainfo del grupo.
- Si dice "ahora" o "ahora mismo", usa la fecha de publicación.
- Si dice "ayer" o "anoche", resta un día.
- Lugar: tal cual aparece en el texto, en minúsculas.
- Devuelve SOLO el JSON, sin markdown."""


def text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def load_cache() -> dict:
    if CACHE_PATH.exists():
        return json.loads(CACHE_PATH.read_text())
    return {}


def save_cache(cache: dict) -> None:
    CACHE_PATH.write_text(json.dumps(cache, ensure_ascii=False, indent=2))


async def extract_one(text: str, pub_date: str) -> dict:
    user_prompt = f"Fecha de publicación: {pub_date}\n\nMensaje:\n{text}"
    resp = await client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        response_format={"type": "json_object"},
        temperature=0.0,
    )
    return json.loads(resp.choices[0].message.content)


async def process_msg(msg: dict, cache: dict, sem: asyncio.Semaphore) -> dict | None:
    text = (msg.get("text") or "").strip()
    if not text:
        return None
    h = text_hash(text)

    if h in cache:
        extracted = cache[h]
    else:
        async with sem:
            try:
                extracted = await extract_one(text, msg["date"])
            except Exception as e:
                print(f"[extract] error msg {msg.get('id')}: {e}")
                return None
        cache[h] = extracted

    return {
        "message_id": msg.get("id"),
        "pub_date": msg.get("date"),
        "text": text,
        **extracted,
    }


async def main(limit: int | None, concurrency: int) -> None:
    messages = json.loads(IN_PATH.read_text())
    if limit:
        messages = messages[:limit]
    print(f"[extract] mensajes a procesar: {len(messages)} (concurrency={concurrency}, model={MODEL})")

    cache = load_cache()
    sem = asyncio.Semaphore(concurrency)

    tasks = [process_msg(m, cache, sem) for m in messages]

    rows: list[dict] = []
    completed = 0
    save_every = 500

    for fut in tqdm_asyncio.as_completed(tasks, total=len(tasks), desc="extrayendo"):
        result = await fut
        completed += 1
        if result is not None:
            rows.append(result)
        if completed % save_every == 0:
            save_cache(cache)

    save_cache(cache)

    df = pd.DataFrame(rows)
    print(f"[extract] filas totales: {len(df)}")
    if "es_control" in df.columns:
        controls = df[df["es_control"] == True]
        print(f"[extract] eventos de control: {len(controls)}")
        if "tipo_control" in df.columns:
            print(df["tipo_control"].value_counts(dropna=False).to_string())
    df.to_csv(OUT_PATH, index=False)
    print(f"[extract] guardado en {OUT_PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="procesar solo N mensajes (debug)")
    parser.add_argument("--concurrency", type=int, default=8, help="peticiones simultáneas a OpenAI")
    args = parser.parse_args()
    asyncio.run(main(args.limit, args.concurrency))
