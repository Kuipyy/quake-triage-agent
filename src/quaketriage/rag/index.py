"""Embedding index + cosine search over RAG chunks (plain numpy, no vector DB).

Agent tool: ``search_guidelines(SearchInput, index, embedder)``.
The embedder is injectable (a callable list[str] -> float array) so tests need no model download.
"""
from __future__ import annotations

import json
import math
import re
from collections import Counter
from collections.abc import Callable
from pathlib import Path

import numpy as np
from pydantic import BaseModel, Field

from quaketriage.config import get

Embedder = Callable[[list[str]], np.ndarray]


def sentence_transformer_embedder(
    model_name: str | None = None, max_seq_length: int | None = None, prefix: str = ""
) -> Embedder:
    """Lazy-load a sentence-transformers model; returns L2-normalised float32 embeddings.

    ``prefix`` is prepended to every text (e5 models need "query: " / "passage: ").
    """
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name or get("rag.embedding_model"))
    model.max_seq_length = max_seq_length or get("rag.max_seq_length")

    def embed(texts: list[str]) -> np.ndarray:
        return model.encode([prefix + t for t in texts], normalize_embeddings=True,
                            convert_to_numpy=True, show_progress_bar=False).astype("float32")

    return embed


_TOKEN = re.compile(r"\w+", re.UNICODE)


def tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


class BM25:
    """Minimal Okapi BM25 over pre-tokenised documents."""

    def __init__(self, docs: list[list[str]], k1: float = 1.5, b: float = 0.75) -> None:
        self.k1, self.b = k1, b
        self.tf = [Counter(d) for d in docs]
        self.len = np.array([len(d) for d in docs], dtype="float64")
        self.avg = self.len.mean() if len(docs) else 0.0
        df = Counter(t for d in docs for t in set(d))
        n = len(docs)
        self.idf = {t: math.log(1 + (n - c + 0.5) / (c + 0.5)) for t, c in df.items()}

    def scores(self, query: list[str]) -> np.ndarray:
        out = np.zeros(len(self.tf))
        for t in set(query):
            if t not in self.idf:
                continue
            for i, tf in enumerate(self.tf):
                f = tf.get(t, 0)
                if f:
                    out[i] += self.idf[t] * f * (self.k1 + 1) / (
                        f + self.k1 * (1 - self.b + self.b * self.len[i] / self.avg))
        return out


def _ranks(scores: np.ndarray) -> np.ndarray:
    ranks = np.empty(len(scores), dtype="int64")
    ranks[np.argsort(-scores)] = np.arange(1, len(scores) + 1)
    return ranks


def config_embedders() -> tuple[Embedder, Embedder]:
    """(passage embedder, query embedder) for the model configured under ``rag`` in the YAML."""
    name, length = get("rag.embedding_model"), get("rag.max_seq_length")
    return (sentence_transformer_embedder(name, length, get("rag.passage_prefix")),
            sentence_transformer_embedder(name, length, get("rag.query_prefix")))


class SearchInput(BaseModel):
    """Input schema of the guideline-search tool."""

    query: str = Field(min_length=3)
    top_k: int = Field(default_factory=lambda: get("rag.top_k"), ge=1, le=20)


class RagIndex:
    def __init__(self, chunks: list[dict], embeddings: np.ndarray) -> None:
        if len(chunks) != len(embeddings):
            raise ValueError("chunks and embeddings must have the same length")
        self.chunks = chunks
        self.embeddings = embeddings.astype("float32")
        self._bm25: BM25 | None = None

    @classmethod
    def build(cls, chunks: list[dict], embed: Embedder, batch_size: int = 32) -> RagIndex:
        vecs = [embed([c["text"] for c in chunks[i:i + batch_size]])
                for i in range(0, len(chunks), batch_size)]
        return cls(chunks, np.vstack(vecs))

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        np.save(directory / "embeddings.npy", self.embeddings)
        with open(directory / "chunks.jsonl", "w", encoding="utf-8") as f:
            for c in self.chunks:
                f.write(json.dumps(c, ensure_ascii=False) + "\n")

    @classmethod
    def load(cls, directory: Path) -> RagIndex:
        with open(directory / "chunks.jsonl", encoding="utf-8") as f:
            chunks = [json.loads(line) for line in f]
        return cls(chunks, np.load(directory / "embeddings.npy"))

    def search(self, query_vec: np.ndarray, k: int = 5) -> list[dict]:
        """Top-k chunks by cosine similarity (embeddings are assumed L2-normalised)."""
        scores = self.embeddings @ query_vec.astype("float32")
        order = np.argsort(-scores)[:k]
        return [{**self.chunks[i], "score": float(scores[i])} for i in order]


    def search_hybrid(
        self, query: str, query_vec: np.ndarray, k: int = 5, rrf_k: int = 60
    ) -> list[dict]:
        """Reciprocal-rank fusion of dense cosine ranking and BM25 ranking."""
        if self._bm25 is None:
            self._bm25 = BM25([tokenize(c["text"]) for c in self.chunks])
        dense = self.embeddings @ query_vec.astype("float32")
        lexical = self._bm25.scores(tokenize(query))
        fused = 1.0 / (rrf_k + _ranks(dense)) + 1.0 / (rrf_k + _ranks(lexical))
        order = np.argsort(-fused)[:k]
        return [{**self.chunks[i], "score": float(fused[i])} for i in order]


def search_guidelines(
    inp: SearchInput, index: RagIndex, embed: Embedder, hybrid: bool = False
) -> list[dict]:
    """Tool entry point: embed the query and return the top-k chunks with scores."""
    qv = embed([inp.query])[0]
    if hybrid:
        return index.search_hybrid(inp.query, qv, inp.top_k)
    return index.search(qv, inp.top_k)
