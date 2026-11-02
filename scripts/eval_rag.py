"""Retrieval accuracy on the gold question set (configs/rag_golden.json): recall@k and MRR.

Usage:
    uv run python scripts/eval_rag.py [--index data/processed/rag]
Writes reports/metrics_rag.json.
"""
import argparse
import json
from pathlib import Path

from quaketriage.config import get
from quaketriage.provenance import write_json
from quaketriage.rag.index import RagIndex, config_embedders

ROOT = Path(__file__).parents[1]


def is_hit(chunk: dict, question: dict) -> bool:
    return any(
        chunk["doc"] == g["doc"]
        and any(chunk["page_start"] <= p <= chunk["page_end"] for p in g["pages"])
        for g in question["gold"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", default=str(ROOT / get("paths.data_processed") / "rag"))
    ap.add_argument("--gold", default=str(ROOT / "configs/rag_golden.json"))
    args = ap.parse_args()

    questions = json.load(open(args.gold, encoding="utf-8"))["questions"]
    index = RagIndex.load(Path(args.index))
    _, embed_query = config_embedders()
    hybrid = get("rag.hybrid")
    ks = (1, 3, 5)
    hits = {k: 0 for k in ks}
    rr, rows = 0.0, []
    for g in questions:
        qv = embed_query([g["q"]])[0]
        top = (index.search_hybrid(g["q"], qv, max(ks)) if hybrid
               else index.search(qv, max(ks)))
        rank = next((i + 1 for i, c in enumerate(top) if is_hit(c, g)), None)
        for k in ks:
            hits[k] += rank is not None and rank <= k
        rr += 1 / rank if rank else 0.0
        rows.append({"q": g["q"], "rank": rank,
                     "top1": f'{top[0]["doc"]} p{top[0]["page_start"]}-{top[0]["page_end"]}'})
        print(f"rank={rank}  {g['q'][:70]}")
    n = len(questions)
    result = {"n_questions": n, "n_chunks": len(index.chunks),
              **{f"recall_at_{k}": round(hits[k] / n, 3) for k in ks}, "mrr": round(rr / n, 3),
              "embedding_model": get("rag.embedding_model"), "hybrid": hybrid,
              "chunk_chars": get("rag.chunk_chars"),
              "per_question": rows}
    print({k: v for k, v in result.items() if k != "per_question"})
    write_json(ROOT / get("paths.reports") / "metrics_rag.json", result)


if __name__ == "__main__":
    main()
