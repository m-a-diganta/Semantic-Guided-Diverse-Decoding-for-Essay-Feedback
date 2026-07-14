"""Unified feedback-generation pipeline.

Exposes four decoding methods behind one interface so every comparison is
fair: same prompt, same model, same `k`, same `max_new_tokens`. The only
thing that varies between runs is the decoding strategy.

Decoding modes:
    greedy        — deterministic, k=1 (reference baseline)
    temperature   — nucleus / top-k sampling with num_return_sequences=k
    diverse_beam  — group beam search with diversity_penalty>0, no SemDiD patch
    semdid        — diverse beam search with SemDiD's semantic repulsion patch

Usage:
    from feedback_pipeline import FeedbackPipeline, DecodingMode
    pipe = FeedbackPipeline(model_name="Qwen/Qwen2.5-0.5B-Instruct",
                            mode=DecodingMode.SEMDID,
                            semdid_module_path="/content/SemDiD/my_lm_eval/lm_eval/models/semantic_search.py")
    outs = pipe.generate(essay_text="…", k=3)
    pipe.save_json(outs, "results/semdid_essay001.json")

The heavy work (monkey-patching, loading VLLM + HF copies of the model,
handling SemDiD's list-of-strings return) lives in the `semdid` code path.
The other three paths are plain HuggingFace.
"""

from __future__ import annotations

import importlib.util
import json
import time
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig


# ---- Public types ---------------------------------------------------------


class DecodingMode(str, Enum):
    GREEDY = "greedy"
    TEMPERATURE = "temperature"
    DIVERSE_BEAM = "diverse_beam"
    SEMDID = "semdid"


@dataclass
class GenerationResult:
    """Single generation call output. Serialisable to JSON for later eval."""
    essay_id: str
    mode: str
    model_name: str
    variants: List[str]
    k: int
    latency_s: float
    prompt: str
    config: Dict[str, Any]

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False, indent=2)


# ---- Prompt construction --------------------------------------------------


FEEDBACK_SYSTEM_PROMPT = (
    "You are an experienced writing teacher providing constructive, specific, "
    "actionable feedback to a student on their essay."
)


def build_feedback_prompt(
    essay: str,
    rubric: Optional[str] = None,
    focus_dimensions: Optional[List[str]] = None,
) -> List[Dict[str, str]]:
    """Build a chat-format message list. Same prompt template across all methods."""
    focus = (
        focus_dimensions
        if focus_dimensions is not None
        else ["argument development", "evidence and support", "organisation", "mechanics"]
    )
    user_msg = (
        "Please give feedback on the following student essay. "
        f"Comment on {', '.join(focus)}. "
        "Write a single concise feedback paragraph (4-6 sentences).\n\n"
    )
    if rubric:
        user_msg += f"Rubric:\n{rubric}\n\n"
    user_msg += f"Essay:\n{essay}\n\nFeedback:"

    return [
        {"role": "system", "content": FEEDBACK_SYSTEM_PROMPT},
        {"role": "user", "content": user_msg},
    ]


# ---- Pipeline -------------------------------------------------------------


class FeedbackPipeline:
    """Unified generation pipeline over four decoding modes."""

    def __init__(
        self,
        model_name: str,
        mode: DecodingMode,
        *,
        device: str = "cuda",
        dtype: torch.dtype = torch.float16,
        semdid_module_path: Optional[str] = None,
    ):
        self.model_name = model_name
        self.mode = DecodingMode(mode)
        self.device = device
        self.dtype = dtype

        if self.mode is DecodingMode.SEMDID:
            if semdid_module_path is None:
                raise ValueError(
                    "DecodingMode.SEMDID requires semdid_module_path pointing to "
                    "the fork's semantic_search.py (e.g. /content/SemDiD/…/semantic_search.py)."
                )
            self._apply_semdid_patch(semdid_module_path)

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            torch_dtype=dtype,
            trust_remote_code=True,
        )
        if self.mode is DecodingMode.SEMDID:
            # Flag SemDiD's monkey-patch checks for
            self.model.use_semantic_diverse_beam_search = True

    @staticmethod
    def _apply_semdid_patch(path: str) -> None:
        """Load SemDiD's semantic_search.py via importlib (no sys.path pollution)."""
        spec = importlib.util.spec_from_file_location("semdid_patch", path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Could not load SemDiD module from {path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        module.patch_transformers_generation()

    def _build_inputs(self, essay: str, rubric: Optional[str]) -> tuple[torch.Tensor, str]:
        messages = build_feedback_prompt(essay, rubric=rubric)
        prompt_text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        input_ids = self.tokenizer(prompt_text, return_tensors="pt").input_ids.to(self.model.device)
        return input_ids, prompt_text

    def _gen_config(self, k: int, max_new_tokens: int, prompt_len: int) -> GenerationConfig:
        """Decoding-mode-specific GenerationConfig. All other params held constant."""
        base = dict(
            max_length=prompt_len + max_new_tokens,
            num_return_sequences=k,
            pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            repetition_penalty=1.2,
        )
        if self.mode is DecodingMode.GREEDY:
            return GenerationConfig(**base, num_beams=1, do_sample=False)
        if self.mode is DecodingMode.TEMPERATURE:
            return GenerationConfig(
                **base,
                num_beams=1,
                do_sample=True,
                temperature=1.0,
                top_p=0.9,
                top_k=50,
            )
        if self.mode is DecodingMode.DIVERSE_BEAM:
            return GenerationConfig(
                **base,
                num_beams=k * 3,
                num_beam_groups=k,
                diversity_penalty=1.0,
            )
        if self.mode is DecodingMode.SEMDID:
            return GenerationConfig(
                **base,
                num_beams=k * 3,
                num_beam_groups=k,
                diversity_penalty=1.0,
                forward_steps=300,  # SemDiD-specific lookahead
            )
        raise ValueError(f"Unknown decoding mode: {self.mode}")

    def generate(
        self,
        essay: str,
        *,
        essay_id: str = "anon",
        k: int = 3,
        max_new_tokens: int = 300,
        rubric: Optional[str] = None,
    ) -> GenerationResult:
        """Generate k feedback variants for one essay."""
        if self.mode is DecodingMode.GREEDY and k != 1:
            # Greedy is deterministic — k>1 would just duplicate.
            k = 1

        input_ids, prompt_text = self._build_inputs(essay, rubric)
        gen_config = self._gen_config(k=k, max_new_tokens=max_new_tokens, prompt_len=input_ids.shape[1])

        start = time.time()
        with torch.inference_mode():
            outputs = self.model.generate(input_ids, generation_config=gen_config)
        latency = time.time() - start

        # SemDiD returns a list of strings; HF returns a (k, seq_len) tensor.
        if isinstance(outputs, list) and all(isinstance(x, str) for x in outputs):
            variants = [s.strip() for s in outputs]
        else:
            variants = [
                self.tokenizer.decode(out[input_ids.shape[1]:], skip_special_tokens=True).strip()
                for out in outputs
            ]

        return GenerationResult(
            essay_id=essay_id,
            mode=self.mode.value,
            model_name=self.model_name,
            variants=variants,
            k=k,
            latency_s=latency,
            prompt=prompt_text,
            config=gen_config.to_dict(),
        )

    @staticmethod
    def save_json(result: GenerationResult, path: str | Path) -> None:
        """Save a GenerationResult as JSON — single source of truth for eval."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(result.to_json(), encoding="utf-8")

    @staticmethod
    def load_json(path: str | Path) -> GenerationResult:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return GenerationResult(**data)
