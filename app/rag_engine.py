"""
Motor RAG de Anti Hog con:
  1. Filtrado por metadatos (hybrid search): parsea zona/tipo/año de la pregunta.
  2. Routing: preguntas estadísticas → pandas; narrativas → RAG.
  3. Re-ranking con cross-encoder multilingüe.
  4. Embeddings e5 (si el índice v2 existe) con prefijos query:/passage:.
  5. Dedup de resultados recuperados.

Diseñado sin dependencias de Streamlit para poder testearse en aislado.
"""

from __future__ import annotations

import os
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
EVENTS_PATH = ROOT / "data" / "processed" / "events_geocoded.csv"
CHROMA_DIR = ROOT / "rag" / "chroma_db"

COLLECTION_V2 = "antihog_v2"          # e5, embeddings enriquecidos
COLLECTION_V1 = "antihog_messages"    # MiniLM (fallback)

E5_MODEL = "intfloat/multilingual-e5-base"
MINILM_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
RERANK_MODEL = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"  # multilingüe (incl. español)

# ── Léxicos para parseo de la pregunta ──────────────────────────────────────
LUGAR_A_ZONA = {
    "arrecife": "capital", "playa honda": "capital", "san bartolome": "capital",
    "aeropuerto": "capital", "tahiche": "capital", "argana": "capital",
    "tias": "sur", "puerto del carmen": "sur", "macher": "sur", "puerto calero": "sur",
    "yaiza": "yaiza", "playa blanca": "yaiza", "femes": "yaiza", "el golfo": "yaiza",
    "tinajo": "interior", "mancha blanca": "interior", "la geria": "interior", "la santa": "interior",
    "teguise": "centro", "costa teguise": "centro", "tao": "centro", "tiagua": "centro",
    "munique": "centro", "soo": "centro", "famara": "centro", "nazaret": "centro",
    "haria": "norte", "orzola": "norte", "punta mujeres": "norte", "mala": "norte",
    "arrieta": "norte", "guatiza": "norte",
}
ZONAS = {"capital", "sur", "yaiza", "interior", "centro", "norte"}

TIPO_KEYWORDS = {
    "alcoholemia": ["alcohol", "alcoholemia"],
    "radar": ["radar", "radares", "velocidad"],
    "drogas": ["droga", "drogas", "estupefaciente"],
    "trafico": ["trafico", "tráfico", "circulacion"],
    "documentacion": ["documentacion", "documentación", "papeles", "carnet", "carné"],
}

STAT_TRIGGERS = [
    "cuanto", "cuantos", "cuantas", "cuántos", "cuántas", "cuánto",
    "que dia", "qué día", "que día", "qué dia", "dia de la semana", "día de la semana",
    "cual zona", "cuál zona", "que zona", "qué zona", "zona con mas", "zona con más",
    "a que hora", "a qué hora", "que hora", "qué hora", "franja",
    "tipo mas", "tipo más", "mas comun", "más común", "frecuente", "frecuencia",
    "porcentaje", "media", "promedio", "total de", "numero de", "número de",
    "cuantos controles", "ranking",
]

DIAS = ["lunes", "martes", "miércoles", "jueves", "viernes", "sábado", "domingo"]


def _norm(s: str) -> str:
    s = (s or "").lower()
    s = "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c))
    return s


@dataclass
class QueryFilters:
    zona: str | None = None
    tipo: str | None = None
    anio: int | None = None
    where: dict = field(default_factory=dict)


@dataclass
class RAGResult:
    answer: str
    sources: list[dict]
    route: str  # "estadistico" | "narrativo"
    filters: QueryFilters


class RAGEngine:
    def __init__(self):
        self.collection = None
        self.embedder = None
        self.reranker = None
        self.use_e5 = False
        self.events = pd.read_csv(EVENTS_PATH) if EVENTS_PATH.exists() else pd.DataFrame()
        if not self.events.empty:
            self.events["fecha_evento"] = pd.to_datetime(
                self.events["fecha"].fillna(self.events["pub_date"]),
                errors="coerce", utc=True).dt.tz_localize(None)
        self._load_collection()

    # ── Carga perezosa ──────────────────────────────────────────────────────
    def _load_collection(self):
        from chromadb import PersistentClient
        from sentence_transformers import SentenceTransformer
        client = PersistentClient(path=str(CHROMA_DIR))
        names = [c.name for c in client.list_collections()]
        if COLLECTION_V2 in names:
            self.collection = client.get_collection(COLLECTION_V2)
            self.embedder = SentenceTransformer(E5_MODEL)
            self.use_e5 = True
        elif COLLECTION_V1 in names:
            self.collection = client.get_collection(COLLECTION_V1)
            self.embedder = SentenceTransformer(MINILM_MODEL)
            self.use_e5 = False
        else:
            raise RuntimeError("No hay índice RAG. Corre scripts/05_build_rag.py")

    def _get_reranker(self):
        if self.reranker is None:
            from sentence_transformers import CrossEncoder
            self.reranker = CrossEncoder(RERANK_MODEL)
        return self.reranker

    def _embed_query(self, q: str):
        text = f"query: {q}" if self.use_e5 else q
        return self.embedder.encode([text]).tolist()

    # ── 1. Parseo de filtros ─────────────────────────────────────────────────
    def parse_query(self, q: str) -> QueryFilters:
        n = _norm(q)
        f = QueryFilters()
        # zona directa
        for z in ZONAS:
            if z in n:
                f.zona = z
        # lugar → zona
        if f.zona is None:
            for lugar, z in LUGAR_A_ZONA.items():
                if lugar in n:
                    f.zona = z
                    break
        # tipo
        for tipo, kws in TIPO_KEYWORDS.items():
            if any(_norm(k) in n for k in kws):
                f.tipo = tipo
                break
        # año
        m = re.search(r"\b(20\d{2})\b", n)
        if m:
            f.anio = int(m.group(1))
        # where para chroma
        where = {}
        if f.zona:
            where["zona"] = f.zona
        if f.tipo:
            where["tipo_control"] = f.tipo
        if len(where) > 1:
            f.where = {"$and": [{k: v} for k, v in where.items()]}
        elif where:
            f.where = where
        return f

    # ── 2. Routing ────────────────────────────────────────────────────────────
    def is_statistical(self, q: str) -> bool:
        n = _norm(q)
        return any(t in n for t in (_norm(x) for x in STAT_TRIGGERS))

    def answer_statistical(self, q: str, f: QueryFilters) -> str:
        if self.events.empty:
            return "No hay datos estructurados disponibles."
        df = self.events[self.events["es_control"] == True].copy()
        if f.zona:
            df = df[df["zona"] == f.zona]
        if f.tipo:
            df = df[df["tipo_control"] == f.tipo]
        if f.anio:
            df = df[df["fecha_evento"].dt.year == f.anio]

        n = _norm(q)
        scope = []
        if f.zona: scope.append(f"zona {f.zona}")
        if f.tipo: scope.append(f"tipo {f.tipo}")
        if f.anio: scope.append(str(f.anio))
        scope_str = (" (" + ", ".join(scope) + ")") if scope else ""

        if df.empty:
            return f"No se encontraron controles{scope_str}."

        # ¿Qué día de la semana?
        if "dia" in n or "día" in n:
            dist = df["fecha_evento"].dt.dayofweek.value_counts().sort_index()
            top = int(dist.idxmax())
            return (f"El día con más controles{scope_str} es **{DIAS[top]}** "
                    f"({int(dist.max())} de {len(df)} controles). "
                    f"Distribución: " + ", ".join(f"{DIAS[i]}={int(dist.get(i,0))}" for i in range(7)) + ".")
        # ¿Qué hora / franja?
        if "hora" in n or "franja" in n:
            dist = df["fecha_evento"].dt.hour.value_counts().sort_index()
            top = int(dist.idxmax())
            return (f"La hora con más reportes{scope_str} es **{top:02d}:00h** "
                    f"({int(dist.max())} controles).")
        # ¿Qué zona con más?
        if ("zona" in n) and ("mas" in n or "más" in n or "cual" in n or "cuál" in n) and not f.zona:
            dist = df["zona"].value_counts()
            return (f"La zona con más controles{scope_str} es **{dist.idxmax()}** "
                    f"({int(dist.max())} de {len(df)}). "
                    f"Ranking: " + ", ".join(f"{z}={c}" for z, c in dist.items()) + ".")
        # ¿Qué tipo más común?
        if "tipo" in n and ("mas" in n or "más" in n or "comun" in n or "común" in n) and not f.tipo:
            dist = df["tipo_control"].value_counts()
            return (f"El tipo de control más común{scope_str} es **{dist.idxmax()}** "
                    f"({int(dist.max())} de {len(df)}).")
        # Por defecto: conteo total
        return f"Se registraron **{len(df)} controles**{scope_str} en el histórico."

    # ── 3+4+5. Retrieval + dedup + rerank ────────────────────────────────────
    def _dedup(self, docs: list[str], metas: list[dict], keep: int) -> tuple[list[str], list[dict]]:
        seen, out_d, out_m = set(), [], []
        for d, m in zip(docs, metas):
            key = _norm(d)[:60]
            if key in seen:
                continue
            seen.add(key)
            out_d.append(d); out_m.append(m)
            if len(out_d) >= keep:
                break
        return out_d, out_m

    def retrieve(self, q: str, f: QueryFilters, n_fetch: int = 30) -> tuple[list[str], list[dict]]:
        kwargs = {"query_embeddings": self._embed_query(q), "n_results": n_fetch}
        if f.where:
            kwargs["where"] = f.where
        try:
            res = self.collection.query(**kwargs)
        except Exception:
            # si el filtro deja 0 resultados o falla, reintenta sin filtro
            res = self.collection.query(query_embeddings=self._embed_query(q), n_results=n_fetch)
        docs = res["documents"][0] if res["documents"] else []
        metas = res["metadatas"][0] if res["metadatas"] else []
        if not docs:  # filtro demasiado estricto → sin filtro
            res = self.collection.query(query_embeddings=self._embed_query(q), n_results=n_fetch)
            docs = res["documents"][0] if res["documents"] else []
            metas = res["metadatas"][0] if res["metadatas"] else []
        return docs, metas

    def rerank(self, q: str, docs: list[str], metas: list[dict], top_k: int = 8):
        if not docs:
            return docs, metas
        ce = self._get_reranker()
        scores = ce.predict([(q, d) for d in docs])
        order = sorted(range(len(docs)), key=lambda i: -scores[i])[:top_k]
        return [docs[i] for i in order], [metas[i] for i in order]

    # ── Orquestación ──────────────────────────────────────────────────────────
    def answer(self, q: str, openai_client=None, model="gpt-4o-mini") -> RAGResult:
        f = self.parse_query(q)

        if self.is_statistical(q):
            ans = self.answer_statistical(q, f)
            return RAGResult(answer=ans, sources=[], route="estadistico", filters=f)

        # narrativo: retrieve → dedup → rerank → LLM
        docs, metas = self.retrieve(q, f, n_fetch=30)
        docs, metas = self._dedup(docs, metas, keep=20)
        docs, metas = self.rerank(q, docs, metas, top_k=8)
        sources = [{"text": d, **m} for d, m in zip(docs, metas)]

        if openai_client is None:
            return RAGResult(answer="(sin LLM) documentos recuperados.", sources=sources, route="narrativo", filters=f)

        context = "\n".join(
            f"[{m.get('date','?')} | {m.get('zona','?')} | {m.get('tipo_control','?')}] {d}"
            for d, m in zip(docs, metas))
        resp = openai_client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": (
                    "Eres un asistente que responde sobre controles policiales históricos en Lanzarote "
                    "basándote SOLO en los mensajes recuperados. Sé conciso. CITA fecha y lugar de cada "
                    "afirmación. Si la info no es suficiente, dilo claramente.")},
                {"role": "user", "content": f"Mensajes:\n{context}\n\nPregunta: {q}"},
            ], temperature=0.2)
        return RAGResult(answer=resp.choices[0].message.content, sources=sources, route="narrativo", filters=f)
