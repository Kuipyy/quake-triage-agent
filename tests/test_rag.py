"""Unit tests for RAG chunking and index (no model download)."""
import numpy as np
import pytest

from quaketriage.rag.chunk import chunk_pages, clean_lines, heading_of, is_toc_page
from quaketriage.rag.index import RagIndex, SearchInput, search_guidelines


def test_heading_detection():
    assert heading_of("2.5.1. Operasyon Servisi") == "2.5.1 Operasyon Servisi"
    assert heading_of("5.7 Assessment, Search and Rescue Levels") == (
        "5.7 Assessment, Search and Rescue Levels")
    assert heading_of("6 Medical") == "6 Medical"
    assert heading_of("1. Completed Worksite Reports.") is None      # list item
    assert heading_of("5.5.1 Why to create sectors ........ 24") is None  # TOC leader
    assert heading_of("Plain sentence without number") is None


def test_clean_lines_strips_boilerplate_and_page_number():
    text = "12 \nUnited Nations Office for the Coordination of Humanitarians Affairs (OCHA) \n" \
           "www.unocha.org \nReal content line.\n"
    assert clean_lines(text) == ["Real content line."]
    tamp = "TAMP TÜRKİYE AFET MÜDAHALE PLANI \n \nwww.afad.gov.tr 25 \nİçerik satırı.\n"
    assert clean_lines(tamp) == ["İçerik satırı."]


def test_toc_page_detected():
    assert is_toc_page("\n".join(f"{i} Title ........ {i}" for i in range(6)))
    assert not is_toc_page("normal text")


def _pages():
    long_line = "Sentence about rescue operations and coordination. " * 3
    return [
        {"doc": "D", "page": 1, "text": f"1 Intro\n{long_line}\n{long_line}\n{long_line}\n"},
        {"doc": "D", "page": 2, "text": f"{long_line}\n2 Second Section\n{long_line}\n"},
    ]


def test_chunks_respect_sections_pages_and_size():
    chunks = chunk_pages(_pages(), chunk_chars=300, overlap_chars=60)
    assert chunks and all(c["doc"] == "D" for c in chunks)
    assert {c["section"] for c in chunks} == {"1 Intro", "2 Second Section"}
    for c in chunks:
        assert c["page_start"] <= c["page_end"]
        if c["section"] == "2 Second Section":
            assert "Second Section" in c["text"] or c["page_start"] == 2
    # no chunk text is repeated verbatim (no overlap-only duplicates)
    assert len({c["text"] for c in chunks}) == len(chunks)
    # the section-1 content that continues on page 2 stays in section 1
    assert any(c["section"] == "1 Intro" and c["page_end"] == 2 for c in chunks)


def test_chunk_param_validation():
    with pytest.raises(ValueError):
        chunk_pages(_pages(), chunk_chars=100, overlap_chars=100)


def _fake_embed(texts):
    vocab = ["rescue", "water", "medical"]
    v = np.array([[t.lower().count(w) for w in vocab] for t in texts], dtype="float32")
    return v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-9)


def test_index_search_roundtrip(tmp_path):
    chunks = [{"doc": "D", "section": "s", "page_start": i, "page_end": i, "text": t}
              for i, t in enumerate(["rescue rescue teams", "water supply water", "medical aid"])]
    idx = RagIndex.build(chunks, _fake_embed, batch_size=2)
    idx.save(tmp_path)
    loaded = RagIndex.load(tmp_path)
    hits = search_guidelines(SearchInput(query="need water", top_k=2), loaded, _fake_embed)
    assert hits[0]["text"] == "water supply water" and hits[0]["score"] > hits[1]["score"]
    assert len(hits) == 2


def test_length_mismatch_raises():
    with pytest.raises(ValueError):
        RagIndex([{"text": "a"}], np.zeros((2, 3)))


def test_list_items_are_not_headings():
    assert heading_of("2. USAR Teams may or may not decide to move") is None
    assert heading_of("2. MÜDAHALE ORGANİZASYONU") == "2 MÜDAHALE ORGANİZASYONU"
    assert heading_of("2.1. Müdahale Aşamaları") == "2.1 Müdahale Aşamaları"


def test_bm25_and_hybrid_search():
    from quaketriage.rag.index import BM25, tokenize

    docs = ["rescue teams search buildings", "clean water supply", "medical triage point"]
    bm = BM25([tokenize(d) for d in docs])
    assert bm.scores(tokenize("triage"))[2] > 0 and bm.scores(tokenize("triage"))[0] == 0
    chunks = [{"doc": "D", "section": "", "page_start": i, "page_end": i, "text": t}
              for i, t in enumerate(docs)]
    idx = RagIndex.build(chunks, _fake_embed)
    hits = idx.search_hybrid("water supply", _fake_embed(["water supply"])[0], k=2)
    assert hits[0]["text"] == "clean water supply"
