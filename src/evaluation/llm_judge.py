"""LLM-as-judge evaluation.

Scores each feedback variant on three dimensions:
    usefulness     — how useful is the feedback for the student?
    specificity    — does it reference *this* essay's content vs generic advice?
    actionability  — can the student concretely act on the suggestions?

Scale: 1-5 per dimension. The judge is asked to score all k variants for one
(essay, method) in a single call so it sees them comparatively.

Backends: OpenAI-compatible chat completions API. Verified targets:
    - LM Studio (default base_url=http://localhost:1234/v1)
    - vLLM serve, llama.cpp server, Ollama (with /v1 prefix), OpenAI proper

Swap the backend by changing `base_url` and `model_id` — the rest of the
pipeline doesn't care.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol

import httpx


JUDGE_DIMENSIONS: tuple[str, ...] = ("usefulness", "specificity", "actionability")


@dataclass
class VariantScore:
    """Scores for a single feedback variant on the three rubric dimensions."""
    variant_index: int           # 1-based to match the prompt
    usefulness: float
    specificity: float
    actionability: float
    rationale: str = ""

    @property
    def mean(self) -> float:
        return (self.usefulness + self.specificity + self.actionability) / 3.0


@dataclass
class JudgmentResult:
    """Judgment for all k variants of one (essay, method) pair."""
    essay_id: str
    method: str
    judge_model: str
    scores: List[VariantScore] = field(default_factory=list)
    raw_response: str = ""
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "JudgmentResult":
        scores = [VariantScore(**s) for s in d.get("scores", [])]
        return cls(
            essay_id=d["essay_id"],
            method=d["method"],
            judge_model=d["judge_model"],
            scores=scores,
            raw_response=d.get("raw_response", ""),
            error=d.get("error"),
        )


# ---- Prompt template ------------------------------------------------------


JUDGE_PROMPT_TEMPLATE = """You are evaluating multiple feedback responses written by different teachers for the same student essay. Score each variant on three dimensions, scale 1-5 (5 = best), and explain briefly.

DIMENSIONS
- usefulness: how useful is this feedback for helping the student improve?
- specificity: how specifically does it reference content of THIS essay (vs generic writing advice)?
- actionability: how concrete and actionable are the suggested improvements?

ESSAY (assigned holistic score: {essay_score})
\"\"\"
{essay}
\"\"\"

FEEDBACK VARIANTS
{variants_text}

Respond ONLY with a single valid JSON object. Do not include any prose outside the JSON. Schema:
{{
  "scores": [
    {{"variant": 1, "usefulness": <int 1-5>, "specificity": <int 1-5>, "actionability": <int 1-5>, "rationale": "<1-2 sentences>"}},
    {{"variant": 2, ...}}
  ]
}}
"""


# ---- Judge interface ------------------------------------------------------


class LLMJudge(Protocol):
    model_id: str

    def judge(
        self,
        *,
        essay: str,
        essay_score: int,
        variants: List[str],
        essay_id: str = "",
        method: str = "",
    ) -> JudgmentResult: ...


# ---- OpenAI-compatible HTTP backend (LM Studio, vLLM serve, etc) ---------


class OpenAICompatibleJudge:
    """Judge using any service that implements POST /v1/chat/completions.

    Constructor defaults match LM Studio's out-of-the-box configuration.
    """

    def __init__(
        self,
        base_url: str = "http://localhost:1234/v1",
        model_id: str = "qwen3.6-35b-a3b-ud-mlx",
        api_key: str = "not-needed",
        timeout: float = 240.0,
        cache_dir: Optional[str | Path] = None,
        max_tokens: int = 4000,
        temperature: float = 0.0,
        disable_thinking: bool = True,
    ):
        self.base_url = base_url.rstrip("/")
        self.model_id = model_id
        self.api_key = api_key
        self.timeout = timeout
        self.max_tokens = max_tokens
        self.temperature = temperature
        # Qwen3-family reasoning models emit <think>...</think> before the
        # answer. LM Studio strips the thinking from the response, so if the
        # model exhausts max_tokens inside thinking, raw_response comes back
        # as the empty string. Appending /no_think disables the reasoning
        # phase entirely. Safe no-op on non-thinking models.
        self.disable_thinking = disable_thinking
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir is not None:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._client = httpx.Client(timeout=timeout)

    # --- public API ---

    def judge(
        self,
        *,
        essay: str,
        essay_score: int,
        variants: List[str],
        essay_id: str = "",
        method: str = "",
    ) -> JudgmentResult:
        if not variants:
            return JudgmentResult(
                essay_id=essay_id, method=method, judge_model=self.model_id,
                error="no variants supplied",
            )

        cache_key = self._cache_key(essay, essay_score, variants)
        cached = self._load_cache(cache_key)
        if cached is not None:
            # Patch in any new identifying fields (cached results are content-only)
            cached.essay_id = essay_id or cached.essay_id
            cached.method = method or cached.method
            return cached

        prompt = self._build_prompt(essay, essay_score, variants)

        try:
            raw = self._call_chat(prompt)
            scores = self._parse_response(raw, n_variants=len(variants))
            result = JudgmentResult(
                essay_id=essay_id, method=method, judge_model=self.model_id,
                scores=scores, raw_response=raw,
            )
        except Exception as e:
            result = JudgmentResult(
                essay_id=essay_id, method=method, judge_model=self.model_id,
                error=f"{type(e).__name__}: {e}",
            )

        if result.error is None:
            self._save_cache(cache_key, result)
        return result

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "OpenAICompatibleJudge":
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

    # --- internals ---

    def _build_prompt(self, essay: str, essay_score: int, variants: List[str]) -> str:
        variants_text = "\n\n".join(
            f"[{i + 1}]\n{v.strip()}" for i, v in enumerate(variants)
        )
        prompt = JUDGE_PROMPT_TEMPLATE.format(
            essay_score=essay_score,
            essay=essay.strip(),
            variants_text=variants_text,
        )
        if self.disable_thinking:
            prompt = prompt.rstrip() + "\n\n/no_think"
        return prompt

    def _call_chat(self, prompt: str) -> str:
        payload: Dict[str, Any] = {
            "model": self.model_id,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "stream": False,
        }
        # Qwen3 models support disabling reasoning via chat_template_kwargs.
        # LM Studio passes this through to the tokenizer's apply_chat_template.
        # If the backend ignores it, the /no_think suffix in the prompt is the
        # fallback. Both layered for robustness.
        if self.disable_thinking:
            payload["chat_template_kwargs"] = {"enable_thinking": False}

        r = self._client.post(
            f"{self.base_url}/chat/completions",
            json=payload,
            headers={"Authorization": f"Bearer {self.api_key}"},
        )
        r.raise_for_status()
        body = r.json()
        return body["choices"][0]["message"]["content"]

    def _cache_key(self, essay: str, essay_score: int, variants: List[str]) -> str:
        h = hashlib.sha256()
        h.update(self.model_id.encode())
        h.update(str(essay_score).encode())
        h.update(essay.encode())
        for v in variants:
            h.update(b"\x00")  # separator so concatenation isn't ambiguous
            h.update(v.encode())
        return h.hexdigest()[:16]

    def _load_cache(self, key: str) -> Optional[JudgmentResult]:
        if self.cache_dir is None:
            return None
        path = self.cache_dir / f"{key}.json"
        if not path.exists():
            return None
        return JudgmentResult.from_dict(json.loads(path.read_text(encoding="utf-8")))

    def _save_cache(self, key: str, result: JudgmentResult) -> None:
        if self.cache_dir is None:
            return
        path = self.cache_dir / f"{key}.json"
        path.write_text(
            json.dumps(result.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    @staticmethod
    def _parse_response(raw: str, n_variants: int) -> List[VariantScore]:
        """Parse the judge JSON, tolerating markdown fences and extra prose."""
        text = raw.strip()
        # Strip ```json … ``` wrapper if model used one
        if text.startswith("```"):
            parts = text.split("```", 2)
            if len(parts) >= 2:
                text = parts[1]
                if text.lstrip().startswith("json"):
                    text = text.lstrip()[4:]
        # Locate the JSON object: first '{' to last '}'
        start, end = text.find("{"), text.rfind("}")
        if start < 0 or end < 0:
            raise ValueError(f"No JSON object in judge response: {raw[:200]!r}")
        data = json.loads(text[start:end + 1])
        scores_in = data.get("scores", [])
        scores: List[VariantScore] = []
        for s in scores_in:
            scores.append(
                VariantScore(
                    variant_index=int(s["variant"]),
                    usefulness=float(s["usefulness"]),
                    specificity=float(s["specificity"]),
                    actionability=float(s["actionability"]),
                    rationale=str(s.get("rationale", "")),
                )
            )
        if len(scores) != n_variants:
            raise ValueError(
                f"Judge returned {len(scores)} scores; expected {n_variants}"
            )
        return scores
