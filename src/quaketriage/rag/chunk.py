"""Split raw page text (scripts/extract_rag_text.py output) into section-aware chunks.

The source PDFs have no outline, so section headings are detected from numbered heading lines
("2.5.1. Operasyon Servisi", "5.7 Assessment, Search and Rescue Levels", "6 Medical"). Chunks never
cross a heading; each chunk carries doc, section, first/last page.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

# Per-page boilerplate (running headers/footers) of the three source documents.
_BOILERPLATE = [
    re.compile(r"^TAMP TÜRKİYE AFET MÜDAHALE PLANI\s*$"),
    re.compile(r"^www\.afad\.gov\.tr(\s+\d+)?\s*$"),
    re.compile(r"^United Nations Office for the Coordination of Humanitarian.*$"),
    re.compile(r"^www\.unocha\.org\s*$"),
]
_PAGE_NUMBER = re.compile(r"^\d{1,3}$")
_HEADING = re.compile(r"^(\d+(?:\.\d+){0,3})\.?\s+([A-ZÇĞİÖŞÜ][^\n]{2,90})$")


def is_toc_page(text: str) -> bool:
    """Table-of-contents pages are full of dotted leaders; they carry no content."""
    return text.count("....") >= 5


def clean_lines(text: str) -> list[str]:
    """Drop running headers/footers, a leading bare page number and blank lines."""
    lines = [ln.strip() for ln in text.splitlines()]
    out = []
    for ln in lines:
        if not ln or any(p.match(ln) for p in _BOILERPLATE):
            continue
        out.append(re.sub(r"\s+", " ", ln))
    if out and _PAGE_NUMBER.match(out[0]):
        out = out[1:]
    return out


def heading_of(line: str) -> str | None:
    """Return 'num title' if ``line`` looks like a numbered section heading."""
    if "...." in line or line.endswith((".", ",", ";", ":")):
        return None
    m = _HEADING.match(line)
    if not m:
        return None
    num, title = m.group(1), m.group(2)
    # "2. Some sentence" is a list item; only upper-case titles ("2. MÜDAHALE ...") or
    # undotted numbers ("6 Medical") count as top-level headings.
    if "." not in num and line[len(num)] == "." and not title.isupper():
        return None
    return f"{num} {title}"


def chunk_pages(
    pages: list[dict], chunk_chars: int = 700, overlap_chars: int = 120
) -> list[dict]:
    """Chunk one document's pages (dicts with doc, page, text) into section-aware chunks."""
    if not 0 <= overlap_chars < chunk_chars:
        raise ValueError("need 0 <= overlap_chars < chunk_chars")
    chunks: list[dict] = []
    section = ""
    buf: list[tuple[str, int]] = []  # (line, page)
    n_new = 0  # lines added since the last flush (overlap lines do not count)
    doc = pages[0]["doc"] if pages else ""

    def flush(keep_overlap: bool) -> None:
        nonlocal buf, n_new
        if buf and n_new > 0:
            text = " ".join(ln for ln, _ in buf).strip()
            if len(text) >= 40:  # skip stub fragments
                chunks.append({"doc": doc, "section": section, "page_start": buf[0][1],
                               "page_end": buf[-1][1], "text": text})
        kept: list[tuple[str, int]] = []
        if keep_overlap and overlap_chars:
            total = 0
            for item in reversed(buf):
                if total + len(item[0]) > overlap_chars:
                    break
                kept.insert(0, item)
                total += len(item[0]) + 1
        buf, n_new = kept, 0

    for page in pages:
        if is_toc_page(page["text"]):
            continue
        for line in clean_lines(page["text"]):
            head = heading_of(line)
            if head:
                flush(keep_overlap=False)
                section = head
            buf.append((line, page["page"]))
            n_new += 1
            if sum(len(ln) + 1 for ln, _ in buf) >= chunk_chars:
                flush(keep_overlap=True)
    flush(keep_overlap=False)
    return chunks


def load_pages(jsonl_path: Path) -> list[dict]:
    with open(jsonl_path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]
