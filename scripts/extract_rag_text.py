"""Extract raw page text + metadata from the RAG source PDFs.

Output: data/interim/rag_text/<doc>.jsonl  (one JSON object per page:
doc, page, section, text) and data/interim/rag_text/summary.json.
All three PDFs have a text layer, so no OCR is needed.

Usage:
    uv run python scripts/extract_rag_text.py
"""
import json
from pathlib import Path

from pypdf import PdfReader

from quaketriage.config import get

ROOT = Path(__file__).parents[1]
DOCS = {
    "AFAD_TAMP": "AFAD_TAMP.pdf",
    "INSARAG_Vol2_ManB": "INSARAG_Vol2_ManB_Operations.pdf",
    "INSARAG_Vol3_Field": "INSARAG_Vol3_FieldGuide.pdf",
}
MIN_CHARS = 50  # pages with less text are flagged (likely images / blank)


def outline_by_page(reader: PdfReader) -> dict[int, str]:
    """Map 0-based page index → title of the closest preceding outline entry."""
    entries: list[tuple[int, str]] = []

    def walk(items):
        for it in items:
            if isinstance(it, list):
                walk(it)
            else:
                try:
                    entries.append((reader.get_destination_page_number(it), str(it.title)))
                except Exception:
                    continue

    try:
        walk(reader.outline)
    except Exception:
        return {}
    entries.sort()
    mapping, current, k = {}, "", 0
    for page in range(len(reader.pages)):
        while k < len(entries) and entries[k][0] <= page:
            current = entries[k][1]
            k += 1
        mapping[page] = current
    return mapping


def main() -> None:
    src = ROOT / get("paths.data_raw") / "docs"
    out_dir = ROOT / get("paths.data_interim") / "rag_text"
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {}
    for doc, fname in DOCS.items():
        reader = PdfReader(src / fname)
        sections = outline_by_page(reader)
        n_empty = n_chars = 0
        with open(out_dir / f"{doc}.jsonl", "w", encoding="utf-8") as f:
            for i, page in enumerate(reader.pages):
                text = (page.extract_text() or "").strip()
                n_chars += len(text)
                n_empty += len(text) < MIN_CHARS
                row = {"doc": doc, "page": i + 1, "section": sections.get(i, ""), "text": text}
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        summary[doc] = {"pages": len(reader.pages), "low_text_pages": n_empty, "chars": n_chars,
                        "has_outline": bool(sections)}
        print(f"{doc}: {summary[doc]}")
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
