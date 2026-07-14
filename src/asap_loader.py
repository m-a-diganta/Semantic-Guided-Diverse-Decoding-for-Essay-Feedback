"""ASAP 2.0 essay dataset loader.

Data source: https://www.kaggle.com/datasets/lburleigh/asap-2-0/data

The kaggle CSV (`ASAP2_train_sourcetexts.csv`) has 24,728 rows across 7 prompts.
Schema (as of 2026-04-30):

    essay_id                  str   stable per-row id
    score                     int   holistic score (1-5 or 1-6 depending on prompt)
    full_text                 str   the student essay
    assignment                str   the prompt instruction shown to students
    prompt_name               str   one of 7 prompt sets — used as prompt_id
    source_text_1..4          str   source materials students were given (NaN if not used)
    economically_disadvantaged str
    student_disability_status str
    ell_status                str
    race_ethnicity            str
    gender                    str

We deliberately do NOT use the demographic columns. Justified in
Methodology: avoids introducing demographic confounds into either
generation or evaluation. The loader silently drops them.

Project decision (2026-04-30): focus on the "Exploring Venus" prompt set,
sample 100 essays stratified by score. See `default_sample()`.
"""

from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import pandas as pd


# ---- Data class -----------------------------------------------------------


@dataclass
class ASAPEssay:
    essay_id: str
    prompt_id: str          # we use prompt_name as the stable key
    essay: str              # full_text
    scores: Dict[str, Any] = field(default_factory=dict)   # {"holistic": int}
    meta: Dict[str, Any] = field(default_factory=dict)     # assignment, source_texts, n_words

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ASAPEssay":
        return cls(**d)


# ---- Loader ---------------------------------------------------------------


# Columns we actually want to load. We exclude demographic columns entirely.
ESSENTIAL_COLUMNS = [
    "essay_id",
    "score",
    "full_text",
    "assignment",
    "prompt_name",
    "source_text_1",
    "source_text_2",
    "source_text_3",
    "source_text_4",
]

# The default focus prompt for this project (decided 2026-04-30).
DEFAULT_PROMPT = "Exploring Venus"


def load_asap(
    data_path: str | Path,
    *,
    prompt_ids: Optional[Iterable[str]] = None,
    limit: Optional[int] = None,
    max_words: Optional[int] = None,
    seed: int = 0,
) -> List[ASAPEssay]:
    """Load (a subset of) ASAP 2.0 essays from the kaggle CSV.

    Parameters
    ----------
    data_path
        Path to `ASAP2_train_sourcetexts.csv` (or equivalent file with the
        same schema).
    prompt_ids
        Optional iterable of prompt names to keep. If None, all 7 prompts
        are returned. The strings must match `prompt_name` values exactly.
    limit
        Optional cap on number of essays returned (after prompt filtering).
        Sampled deterministically using `seed`. For stratified sampling
        across scores, use `stratified_sample()` instead.
    max_words
        Optional max essay length in whitespace-split words. Drops longer
        essays before applying `limit`.
    seed
        RNG seed for reproducible sampling.

    Returns
    -------
    List[ASAPEssay]
    """
    df = pd.read_csv(data_path, usecols=ESSENTIAL_COLUMNS)

    if prompt_ids is not None:
        prompt_set = set(prompt_ids)
        df = df[df["prompt_name"].isin(prompt_set)].copy()

    if max_words is not None:
        df["__n_words"] = df["full_text"].str.split().str.len()
        df = df[df["__n_words"] <= max_words].copy()

    if limit is not None and len(df) > limit:
        df = df.sample(n=limit, random_state=seed).reset_index(drop=True)

    return [_row_to_essay(row) for _, row in df.iterrows()]


def stratified_sample(
    data_path: str | Path,
    *,
    prompt_id: str,
    per_score: Dict[int, int],
    max_words: Optional[int] = None,
    seed: int = 0,
) -> List[ASAPEssay]:
    """Sample n essays per score tier from a single prompt.

    Example: `per_score={1: 30, 2: 30, 3: 25, 4: 15}` returns 100 essays
    with the given count per holistic score.
    """
    df = pd.read_csv(data_path, usecols=ESSENTIAL_COLUMNS)
    df = df[df["prompt_name"] == prompt_id].copy()

    if max_words is not None:
        df["__n_words"] = df["full_text"].str.split().str.len()
        df = df[df["__n_words"] <= max_words].copy()

    samples = []
    for score, n in per_score.items():
        bucket = df[df["score"] == score]
        if len(bucket) < n:
            raise ValueError(
                f"Requested {n} essays at score {score} but only {len(bucket)} "
                f"available (after max_words filter, prompt={prompt_id!r})."
            )
        samples.append(bucket.sample(n=n, random_state=seed))

    sampled = pd.concat(samples).reset_index(drop=True)
    # Shuffle so consumers don't get score-clustered batches.
    sampled = sampled.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return [_row_to_essay(row) for _, row in sampled.iterrows()]


def default_sample(data_path: str | Path) -> List[ASAPEssay]:
    """The project's canonical 100-essay sample (Exploring Venus, scores 1-4).

    Stable across reruns — uses `seed=0`. Skips scores 5/6 (only 217 essays
    total in those bands across the entire prompt; feedback for high-quality
    work tends to be vague and would muddy the diversity signal).
    """
    return stratified_sample(
        data_path,
        prompt_id=DEFAULT_PROMPT,
        per_score={1: 30, 2: 30, 3: 25, 4: 15},
        max_words=500,
        seed=0,
    )


# ---- Persistence ----------------------------------------------------------


def save_sample(essays: Sequence[ASAPEssay], path: str | Path) -> None:
    """Persist a sample as JSON so eval scripts read from a stable snapshot."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = [e.to_dict() for e in essays]
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def load_sample(path: str | Path) -> List[ASAPEssay]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return [ASAPEssay.from_dict(d) for d in payload]


# ---- Internals ------------------------------------------------------------


def _row_to_essay(row: pd.Series) -> ASAPEssay:
    """Convert a CSV row to an ASAPEssay, dropping NaN source texts."""
    source_texts = [
        row[c] for c in ("source_text_1", "source_text_2", "source_text_3", "source_text_4")
        if isinstance(row[c], str) and row[c].strip()
    ]
    return ASAPEssay(
        essay_id=str(row["essay_id"]),
        prompt_id=str(row["prompt_name"]),
        essay=str(row["full_text"]),
        scores={"holistic": int(row["score"])},
        meta={
            "assignment": str(row["assignment"]),
            "source_texts": source_texts,
            "n_words": len(str(row["full_text"]).split()),
        },
    )


# ---- Fixture (kept for offline pipeline-only smoke-tests) -----------------


def fixture_essays(n: int = 3) -> List[ASAPEssay]:
    """Hand-built fixture for pipeline smoke-testing without the real CSV."""
    essays = [
        ASAPEssay(
            essay_id="fx_001",
            prompt_id="toy_technology",
            essay=(
                "Technology has changed the way we live alot. Many people think phones "
                "are bad but actually they help us communicate with family members who "
                "live far away. Some students use phones in class which is very "
                "distracting to other people. Also social media can be addicting. "
                "Overall I think technology is good because without it we would be "
                "stuck in the stone age and thats not good for anyone."
            ),
            scores={"holistic": 2},
        ),
        ASAPEssay(
            essay_id="fx_002",
            prompt_id="toy_libraries",
            essay=(
                "Libraries are important because people can read books there. They also "
                "have computers and internet for people who don't have it at home. In "
                "conclusion libraries should get more funding because they help alot of "
                "people in the community."
            ),
            scores={"holistic": 1},
        ),
        ASAPEssay(
            essay_id="fx_003",
            prompt_id="toy_school_uniform",
            essay=(
                "School uniforms are a controversial topic. On one hand, uniforms reduce "
                "bullying based on clothing choices and foster a sense of community. On "
                "the other hand, they limit student self-expression. A balanced policy "
                "could allow uniforms with optional accessories, preserving both "
                "equality and individuality."
            ),
            scores={"holistic": 4},
        ),
    ]
    return essays[:n]
