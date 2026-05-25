"""
Evaluación del RAG: mide recall@8 y MRR sobre un set de preguntas con
metadatos esperados (zona / tipo). Compara baseline (cosine puro) vs
hybrid (filtro de metadatos + dedup + re-ranking).

Un documento recuperado se considera "relevante" si su metadata coincide
con la zona y/o tipo esperados de la pregunta.

Uso:
    python scripts/eval_rag.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "app"))

from rag_engine import RAGEngine, QueryFilters  # noqa: E402

# Set de evaluación: (pregunta, zona_esperada, tipo_esperado)
EVAL = [
    ("controles en arrecife", "capital", None),
    ("alcoholemia en playa blanca", "yaiza", "alcoholemia"),
    ("radar en muñique", "centro", "radar"),
    ("guardia civil en puerto del carmen", "sur", None),
    ("control en costa teguise", "centro", None),
    ("alcoholemia en tías", "sur", "alcoholemia"),
    ("radares zona norte haría", "norte", "radar"),
    ("controles en yaiza", "yaiza", None),
    ("droga en arrecife", "capital", "drogas"),
    ("control de documentación en tinajo", "interior", "documentacion"),
    ("policía local en san bartolomé", "capital", None),
    ("radar circunvalación arrecife", "capital", "radar"),
]

K = 8


def relevant(meta: dict, zona, tipo) -> bool:
    ok = True
    if zona is not None:
        ok = ok and (meta.get("zona") == zona)
    if tipo is not None:
        ok = ok and (meta.get("tipo_control") == tipo)
    return ok


def eval_run(engine: RAGEngine, use_hybrid: bool):
    recalls, rrs = [], []
    for q, zona, tipo in EVAL:
        f = engine.parse_query(q) if use_hybrid else QueryFilters()
        docs, metas = engine.retrieve(q, f, n_fetch=30)
        if use_hybrid:
            docs, metas = engine._dedup(docs, metas, keep=20)
            docs, metas = engine.rerank(q, docs, metas, top_k=K)
        else:
            docs, metas = docs[:K], metas[:K]

        hits = [relevant(m, zona, tipo) for m in metas]
        recalls.append(1.0 if any(hits) else 0.0)
        rr = 0.0
        for rank, h in enumerate(hits, 1):
            if h:
                rr = 1.0 / rank
                break
        rrs.append(rr)
    return sum(recalls) / len(recalls), sum(rrs) / len(rrs)


def main():
    print("Cargando motor RAG…")
    engine = RAGEngine()
    print(f"Índice: {'e5 (v2)' if engine.use_e5 else 'MiniLM (v1)'}  |  preguntas: {len(EVAL)}\n")

    print("Evaluando BASELINE (cosine puro, top-8)…")
    r_base, m_base = eval_run(engine, use_hybrid=False)

    print("Evaluando HYBRID (filtro + dedup + rerank)…")
    r_hyb, m_hyb = eval_run(engine, use_hybrid=True)

    print("\n" + "=" * 46)
    print(f"{'':<14}{'recall@8':>14}{'MRR':>14}")
    print("-" * 46)
    print(f"{'baseline':<14}{r_base:>14.3f}{m_base:>14.3f}")
    print(f"{'hybrid':<14}{r_hyb:>14.3f}{m_hyb:>14.3f}")
    print("-" * 46)
    print(f"{'Δ mejora':<14}{r_hyb-r_base:>+14.3f}{m_hyb-m_base:>+14.3f}")
    print("=" * 46)


if __name__ == "__main__":
    main()
