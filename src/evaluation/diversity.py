"""Intrinsic-diversity metrics over feedback variants.

These metrics don't require a gold standard — they measure how different the
k outputs are *from each other*. Useful as a sanity check that a "diverse"
decoding method actually produces diverse outputs, and as one dimension of
the overall evaluation.

Metrics implemented:
    pairwise_cosine(embs)      → full similarity matrix
    mean_pairwise_similarity   → single scalar (lower = more diverse)
    self_bleu(variants)        → TODO — token-level overlap (lower = more diverse)
    distinct_n(variants, n)    → TODO — fraction of unique n-grams

For semantic diversity we use sentence-transformers all-MiniLM-L6-v2 by
default (same as SemDiD uses internally, so we're comparing on a
consistent embedding space).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional, Sequence

import numpy as np


@dataclass
class DiversityScores:
    mean_pairwise_similarity: float
    pairwise_matrix: np.ndarray
    n_variants: int

    def as_dict(self) -> dict:
        return {
            "mean_pairwise_similarity": float(self.mean_pairwise_similarity),
            "pairwise_matrix": self.pairwise_matrix.tolist(),
            "n_variants": int(self.n_variants),
        }


# Module-level cache for the embedding model. Loading it is the expensive
# part — without this, run_local.py was paying ~5s per essay for evaluation
# even when nothing needed embedding (judge call timing was masking this).
_EMBEDDING_MODEL_CACHE: dict[str, Any] = {}


def _get_embedding_model(model_name: str):
    if model_name not in _EMBEDDING_MODEL_CACHE:
        from sentence_transformers import SentenceTransformer  # heavy import: lazy

        _EMBEDDING_MODEL_CACHE[model_name] = SentenceTransformer(model_name)
    return _EMBEDDING_MODEL_CACHE[model_name]


def embed_texts(
    texts: Sequence[str],
    model_name: str = "all-MiniLM-L6-v2",
) -> np.ndarray:
    """Embed a list of texts with sentence-transformers. Normalised."""
    model = _get_embedding_model(model_name)
    return model.encode(list(texts), convert_to_numpy=True, normalize_embeddings=True)


def pairwise_cosine(embeddings: np.ndarray) -> np.ndarray:
    """Cosine similarity matrix for already-normalised embeddings (k x k)."""
    return embeddings @ embeddings.T


def mean_off_diagonal_similarity(matrix: np.ndarray) -> float:
    """Mean of off-diagonal entries of a square similarity matrix."""
    n = matrix.shape[0]
    if n < 2:
        return float("nan")
    off = matrix.sum() - np.trace(matrix)
    return float(off / (n * (n - 1)))


def score(
    variants: Sequence[str],
    model_name: str = "all-MiniLM-L6-v2",
) -> DiversityScores:
    """One-shot: embed, compute matrix, return scores."""
    embs = embed_texts(variants, model_name=model_name)
    sim = pairwise_cosine(embs)
    return DiversityScores(
        mean_pairwise_similarity=mean_off_diagonal_similarity(sim),
        pairwise_matrix=sim,
        n_variants=len(variants),
    )


# TODO (not blocking): self-BLEU and distinct-n implementations for
# token-level diversity. Useful as a complement to embedding-space metrics,
# because SemDiD could in principle produce high semantic diversity with
# low lexical diversity (or vice versa).
