"""Citation-marker normalisation."""
from quaketriage.agent.citations import cited_refs, normalize_citations
from quaketriage.agent.tools import check_support, validate_citations


def test_fullwidth_markers_become_ascii():
    assert normalize_citations("a 【1】 b ［2］ c [3]") == "a [1] b [2] c [3]"
    assert normalize_citations("x 【12†source】") == "x [12]"  # tool-style suffix is dropped


def test_text_without_citations_is_unchanged():
    text = "no marker, but numbers 2023 and (4) stay"
    assert normalize_citations(text) == text


def test_cited_refs_sees_all_marker_styles():
    assert cited_refs("one [1], two 【2】, three ［3］, again [1]") == {1, 2, 3}


def test_validators_accept_fullwidth_markers():
    hits = [{"ref": 1, "doc": "D", "pages": "1-1", "text": "USAR teams register at the RDC"}]
    chunks = [{"doc": "D", "page_start": 1, "page_end": 1,
               "text": "USAR teams register at the RDC"}]
    res = validate_citations("Teams register 【1】 and 【5】.", hits, chunks)
    assert res["cited_refs"] == [1, 5] and res["dangling_refs"] == [5]
    ascii_result = check_support("USAR teams register at the RDC [1].", hits)
    fullwidth_result = check_support("USAR teams register at the RDC 【1】.", hits)
    assert ascii_result and fullwidth_result == ascii_result
