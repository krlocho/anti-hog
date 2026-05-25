"""
Descarga mensajes del canal Telegram y los acumula en un dataset histórico.

IMPORTANTE: el canal tiene auto-delete a las 24h. Por eso este script:
- Guarda CADA scrape como snapshot timestamped (data/raw/snapshots/) — backup append-only.
- Mantiene data/raw/messages.json como dataset acumulado (merge + dedupe por id).
- Sigue siendo incremental (min_id) para minimizar tráfico, pero re-escanea siempre
  el último día por si algún mensaje se publicó antes pero aún está visible.

Uso:
    python scripts/01_scrape_telegram.py            # ejecución normal (interactiva primera vez)
    python scripts/01_scrape_telegram.py --no-auth  # modo headless: falla si la sesión no existe
"""

import argparse
import asyncio
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
from telethon import TelegramClient
from telethon.tl.types import Message

load_dotenv()

API_ID = int(os.environ["TG_API_ID"])
API_HASH = os.environ["TG_API_HASH"]
PHONE = os.environ["TG_PHONE"]
CHANNEL = os.environ["TG_CHANNEL"]

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
SNAP_DIR = RAW_DIR / "snapshots"
RAW_DIR.mkdir(parents=True, exist_ok=True)
SNAP_DIR.mkdir(parents=True, exist_ok=True)

OUT_PATH = RAW_DIR / "messages.json"           # dataset acumulado (canónico)
STATE_PATH = RAW_DIR / "scrape_state.json"     # last_message_id + último run
SESSION_PATH = RAW_DIR / "telegram.session"


def load_existing() -> tuple[list[dict], int | None]:
    if not OUT_PATH.exists():
        return [], None
    with OUT_PATH.open("r", encoding="utf-8") as f:
        messages = json.load(f)
    last_id = None
    if STATE_PATH.exists():
        last_id = json.loads(STATE_PATH.read_text()).get("last_message_id")
    return messages, last_id


def serialize(msg: Message, scraped_at: str) -> dict:
    return {
        "id": msg.id,
        "date": msg.date.isoformat() if msg.date else None,
        "text": msg.message or "",
        "sender_id": getattr(msg.sender_id, "user_id", msg.sender_id),
        "reply_to": msg.reply_to_msg_id,
        "scraped_at": scraped_at,
    }


async def resolve_channel(client: TelegramClient, channel: str):
    """Resuelve TG_CHANNEL como username (@x), ID numérico, URL t.me/, o título exacto entre diálogos."""
    s = channel.strip()
    if s.startswith("@") or s.lstrip("-").isdigit() or s.startswith("https://t.me/"):
        return await client.get_entity(s)

    print(f"[scrape] buscando canal por título: '{s}'")
    matches = []
    async for dialog in client.iter_dialogs():
        if dialog.name and s.lower() in dialog.name.lower():
            matches.append(dialog)

    if not matches:
        print("[scrape] no se encontró ningún diálogo coincidente. Primeros 30 disponibles:")
        async for dialog in client.iter_dialogs(limit=30):
            print(f"  - {dialog.name!r}  (id={dialog.id})")
        raise SystemExit(f"Canal '{s}' no encontrado. Edita TG_CHANNEL con el nombre exacto o el id.")

    if len(matches) > 1:
        print("[scrape] múltiples coincidencias, usando la primera:")
        for d in matches:
            print(f"  - {d.name!r}  (id={d.id})")

    entity = matches[0].entity
    print(f"[scrape] canal resuelto: '{matches[0].name}' (id={matches[0].id})")
    return entity


async def scrape(no_auth: bool) -> None:
    existing, last_id = load_existing()
    print(f"[scrape] mensajes ya guardados: {len(existing)}  last_id={last_id}")

    if no_auth and not SESSION_PATH.exists():
        sys.exit("[scrape] modo --no-auth pero no hay sesión guardada. Ejecuta una vez sin --no-auth para autenticar.")

    client = TelegramClient(str(SESSION_PATH), API_ID, API_HASH)
    await client.start(phone=PHONE)

    entity = await resolve_channel(client, CHANNEL)

    scraped_at = datetime.now(timezone.utc).isoformat()
    new_raw: list[dict] = []
    # Descargamos TODO lo que el canal tiene visible (sin min_id) porque el auto-delete
    # de 24h hace que el riesgo de perder mensajes sea mayor que el de descargar duplicados.
    # La deduplicación posterior se encarga.
    async for msg in client.iter_messages(entity, reverse=True):
        if not msg.message:
            continue
        new_raw.append(serialize(msg, scraped_at))
        if len(new_raw) % 200 == 0:
            print(f"[scrape] leídos {len(new_raw)} del canal…")

    await client.disconnect()

    # Snapshot append-only — backup de seguridad por si la deduplicación falla
    snap_name = scraped_at.replace(":", "-").split(".")[0] + "Z.json"
    snap_path = SNAP_DIR / snap_name
    snap_path.write_text(json.dumps(new_raw, ensure_ascii=False, indent=2))
    print(f"[scrape] snapshot guardado: {snap_path.name}  ({len(new_raw)} msgs)")

    # Merge en dataset canónico (dedupe por id, conservando el primero visto)
    seen_ids = {m["id"] for m in existing}
    truly_new = [m for m in new_raw if m["id"] not in seen_ids]
    merged = existing + truly_new

    OUT_PATH.write_text(json.dumps(merged, ensure_ascii=False, indent=2))

    state = {
        "last_message_id": max((m["id"] for m in merged), default=None),
        "last_scrape": scraped_at,
        "total_messages": len(merged),
        "last_scrape_new": len(truly_new),
        "last_scrape_visible": len(new_raw),
    }
    STATE_PATH.write_text(json.dumps(state, indent=2))

    print(f"[scrape] visibles en canal ahora: {len(new_raw)}")
    print(f"[scrape] nuevos añadidos al histórico: {len(truly_new)}")
    print(f"[scrape] TOTAL acumulado: {len(merged)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-auth", action="store_true", help="falla si la sesión no existe (para cron)")
    args = parser.parse_args()
    asyncio.run(scrape(args.no_auth))
