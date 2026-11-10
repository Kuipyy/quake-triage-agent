"""Citation-marker helpers shared by the agent, the citation validator and the evaluation."""
import re

# Some models write citations as full-width brackets (e.g. 【1】 or ［1］) instead of [1].
_FULLWIDTH = re.compile(r"[【［]\s*(\d+)[^】］\d]*[】］]")
_CITE = re.compile(r"\[(\d+)\]")


def normalize_citations(text: str) -> str:
    """Rewrite full-width citation markers as ``[n]`` so one regex finds every citation."""
    return _FULLWIDTH.sub(lambda m: f"[{m.group(1)}]", text)


def cited_refs(text: str) -> set[int]:
    """Set of citation numbers in ``text`` (after normalisation)."""
    return {int(n) for n in _CITE.findall(normalize_citations(text))}
