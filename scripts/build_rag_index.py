"""Chunk the extracted RAG text and embed it (model from configs/default.yaml, runs locally, free).

Usage:
    uv run python scripts/extract_rag_text.py   # once
    uv run python scripts/build_rag_index.py [--chunk-chars 700 --overlap 120]
Writes data/processed/rag/{chunks.jsonl,embeddings.npy}.
"""
import argparse
import time
from pathlib import Path

from quaketriage.config import get
from quaketriage.rag.chunk import chunk_pages, load_pages
from quaketriage.rag.index import RagIndex, config_embedders

ROOT = Path(__file__).parents[1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunk-chars", type=int, default=get("rag.chunk_chars"))
    ap.add_argument("--overlap", type=int, default=get("rag.chunk_overlap_chars"))
    ap.add_argument("--out", default=str(ROOT / get("paths.data_processed") / "rag"))
    args = ap.parse_args()

    text_dir = ROOT / get("paths.data_interim") / "rag_text"
    chunks = []
    for path in sorted(text_dir.glob("*.jsonl")):
        c = chunk_pages(load_pages(path), args.chunk_chars, args.overlap)
        print(f"{path.stem}: {len(c)} chunks")
        chunks += c
    t0 = time.time()
    embed_passages, _ = config_embedders()
    index = RagIndex.build(chunks, embed_passages)
    index.save(Path(args.out))
    print(f"{len(chunks)} chunks embedded ({index.embeddings.shape[1]}-d) in {time.time()-t0:.0f}s"
          f" → {Path(args.out).relative_to(ROOT)}")


if __name__ == "__main__":
    main()
