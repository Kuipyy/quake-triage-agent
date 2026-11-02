"""Retrieval ablation on the gold set: embedding model x chunk size x dense/hybrid.

All candidates are free, local models. Writes reports/rag_ablation.json.
Usage: uv run python scripts/rag_ablation.py
"""
import json
from pathlib import Path

from quaketriage.config import get
from quaketriage.provenance import write_json
from quaketriage.rag.chunk import chunk_pages, load_pages
from quaketriage.rag.index import RagIndex, sentence_transformer_embedder

ROOT = Path(__file__).parents[1]
MINILM = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
E5 = "intfloat/multilingual-e5-small"
MODELS = {  # name -> (model id, passage prefix, query prefix)
    "minilm": (MINILM, "", ""),
    "e5-small": (E5, "passage: ", "query: "),
}
CHUNK_SIZES = (400, 700)


def is_hit(chunk, question):
    return any(
        chunk["doc"] == g["doc"]
        and any(chunk["page_start"] <= p <= chunk["page_end"] for p in g["pages"])
        for g in question["gold"])


def main() -> None:
    gold = json.load(open(ROOT / "configs/rag_golden.json", encoding="utf-8"))["questions"]
    text_dir = ROOT / get("paths.data_interim") / "rag_text"
    pages = {p.stem: load_pages(p) for p in sorted(text_dir.glob("*.jsonl"))}
    results = []
    for size in CHUNK_SIZES:
        chunks = [c for d in pages.values() for c in chunk_pages(d, size, int(size * 0.17))]
        for name, (mid, pp, qp) in MODELS.items():
            embed_p = sentence_transformer_embedder(mid, 256, pp)
            embed_q = sentence_transformer_embedder(mid, 256, qp)
            index = RagIndex.build(chunks, embed_p)
            for mode in ("dense", "hybrid"):
                ranks = []
                for g in gold:
                    qv = embed_q([g["q"]])[0]
                    top = (index.search(qv, 5) if mode == "dense"
                           else index.search_hybrid(g["q"], qv, 5))
                    ranks.append(next((i + 1 for i, c in enumerate(top) if is_hit(c, g)), None))
                row = {"model": name, "chunk_chars": size, "mode": mode, "n_chunks": len(chunks),
                       "recall_at_1": sum(r == 1 for r in ranks) / len(gold),
                       "recall_at_3": sum(r is not None and r <= 3 for r in ranks) / len(gold),
                       "recall_at_5": sum(r is not None for r in ranks) / len(gold),
                       "mrr": round(sum(1 / r for r in ranks if r) / len(gold), 3), "ranks": ranks}
                results.append(row)
                print({k: v for k, v in row.items() if k != "ranks"}, ranks, flush=True)
    write_json(ROOT / get("paths.reports") / "rag_ablation.json", {"n_questions": len(gold),
                                                                  "results": results})


if __name__ == "__main__":
    main()
