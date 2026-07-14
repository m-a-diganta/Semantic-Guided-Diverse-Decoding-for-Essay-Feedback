"""Rubric-coverage evaluation.

For each feedback variant, detect which of the four pedagogical dimensions
it addresses:

    argument      — claim quality, thesis, position, reasoning
    evidence      — use of source text, examples, support
    organisation  — structure, transitions, flow
    mechanics     — grammar, spelling, punctuation, syntax

This is the rubric-aligned diversity claim: SemDiD should produce variants
that *cover more dimensions* than baseline decoding does.

Implementation: rule-based keyword detection. Cheap, deterministic, easy
to inspect. Keywords were curated from feedback rubrics; they're not
exhaustive but should catch most explicit references. The list can be
extended without changing call sites.

A future upgrade path is an LLM-backed classifier — same
interface, swap the implementation. Rule-based first because it's
zero-cost, fully transparent, and performs well enough on visibly
on-topic feedback.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Sequence


DIMENSIONS: tuple[str, ...] = ("argument", "evidence", "organisation", "mechanics")


# Keyword patterns per dimension. Use word boundaries so "argument" doesn't
# fire on "arguably". Include common British/American spellings.
_KEYWORDS: Dict[str, List[str]] = {
    "argument": [
        r"\bargument\b", r"\barguments\b", r"\bargue", r"\barguing\b",
        r"\bclaim\b", r"\bclaims\b", r"\bthesis\b", r"\bposition\b",
        r"\bstance\b", r"\bperspective\b", r"\breasoning\b", r"\blogic\b",
        r"\blogical\b", r"\bcounter\s*argument", r"\brebuttal\b",
        r"\bevaluat", r"\bassess", r"\bjustif",
    ],
    "evidence": [
        r"\bevidence\b", r"\bsupport(s|ed|ing)?\b", r"\bexample(s)?\b",
        r"\bcite\b", r"\bcitation\b", r"\bquote(s|d)?\b", r"\bsource(s)?\b",
        r"\bdetail(s)?\b", r"\bfact(s)?\b", r"\bdata\b", r"\bspecific\b",
        r"\bback(s|ed|ing)? up\b", r"\billustrat", r"\bdemonstrat",
        r"\bfrom the (article|text|passage|source)",
    ],
    "organisation": [
        # spelling variants
        r"\borganis", r"\borganiz",
        r"\bstructure\b", r"\bstructured\b", r"\bparagraph(s)?\b",
        r"\btransition(s|al)?\b", r"\bflow(s|ing|ed)?\b", r"\bcoheren",
        r"\bcohesion\b", r"\bintroduction\b", r"\bintro\b", r"\bconclusion\b",
        r"\bbody paragraph", r"\btopic sentence", r"\bopening\b", r"\bclosing\b",
        r"\border\b", r"\bsequencing\b",
    ],
    "mechanics": [
        r"\bgrammar\b", r"\bgrammatical\b", r"\bspell(ing|ed)?\b",
        r"\btypo(s)?\b", r"\bpunctuation\b", r"\bcomma(s)?\b", r"\bperiod(s)?\b",
        r"\bsemicolon", r"\bcapital", r"\bsentence (structure|fragment)",
        r"\brun-?on\b", r"\bsubject-verb\b", r"\bagreement\b",
        r"\bapostroph", r"\bsyntax\b", r"\bproofread",
    ],
}

# Pre-compile for speed. Using IGNORECASE.
_PATTERNS: Dict[str, List[re.Pattern[str]]] = {
    dim: [re.compile(p, re.IGNORECASE) for p in pats]
    for dim, pats in _KEYWORDS.items()
}


@dataclass
class CoverageResult:
    """Coverage flags + counts for a single variant."""
    variant_index: int
    flags: Dict[str, bool] = field(default_factory=dict)
    counts: Dict[str, int] = field(default_factory=dict)

    @property
    def n_dimensions_covered(self) -> int:
        return sum(1 for v in self.flags.values() if v)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class CoverageSummary:
    """Aggregate coverage across all variants for one (essay, method)."""
    essay_id: str
    method: str
    per_variant: List[CoverageResult] = field(default_factory=list)

    @property
    def union_dimensions(self) -> List[str]:
        """Dimensions addressed by at least one variant. The headline metric."""
        union = set()
        for r in self.per_variant:
            union.update(d for d, hit in r.flags.items() if hit)
        return sorted(union)

    @property
    def n_union(self) -> int:
        """Single scalar headline: how many dimensions are covered across the k variants."""
        return len(self.union_dimensions)

    def to_dict(self) -> dict:
        return {
            "essay_id": self.essay_id,
            "method": self.method,
            "per_variant": [r.to_dict() for r in self.per_variant],
            "union_dimensions": self.union_dimensions,
            "n_union": self.n_union,
        }


# ---- Public API -----------------------------------------------------------


def coverage_for_variant(text: str, variant_index: int = 0) -> CoverageResult:
    """Detect which dimensions a single feedback string addresses."""
    flags: Dict[str, bool] = {}
    counts: Dict[str, int] = {}
    for dim in DIMENSIONS:
        n = sum(len(p.findall(text)) for p in _PATTERNS[dim])
        counts[dim] = n
        flags[dim] = n > 0
    return CoverageResult(variant_index=variant_index, flags=flags, counts=counts)


def summarise(
    variants: Sequence[str],
    *,
    essay_id: str = "",
    method: str = "",
) -> CoverageSummary:
    """Compute per-variant coverage and the union summary in one call."""
    per_variant = [coverage_for_variant(v, i + 1) for i, v in enumerate(variants)]
    return CoverageSummary(essay_id=essay_id, method=method, per_variant=per_variant)
