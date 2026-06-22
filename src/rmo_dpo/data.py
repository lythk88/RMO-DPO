from __future__ import annotations

import itertools
import random
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import torch
from torch.utils.data import Dataset

from .utils import read_jsonl

HELPSTEER2_OBJECTIVES = ["helpfulness", "correctness", "coherence", "complexity", "verbosity"]
ROLE_MARKER_RE = re.compile(r"<extra_id_1>(User|Assistant)", flags=re.IGNORECASE)


@dataclass
class PairBuildStats:
    split: str
    objective: str
    total_pairs: int
    skipped_equal_score: int
    flipped_by_noise: int


def noise_rates_for_split(split: str, noise_rates: dict[str, float] | None) -> dict[str, float]:
    """Apply preference-label noise only to training data."""
    if split != "train" or not noise_rates:
        return {}
    return dict(noise_rates)


def parse_helpsteer2_dialogue(raw_prompt: str, system_message: str | None = None) -> list[dict[str, str]]:
    """Convert HelpSteer2's lightweight multi-turn format to chat messages."""
    raw_prompt = raw_prompt.strip()
    messages: list[dict[str, str]] = []
    if system_message:
        messages.append({"role": "system", "content": system_message})

    matches = list(ROLE_MARKER_RE.finditer(raw_prompt))
    if not matches:
        messages.append({"role": "user", "content": raw_prompt})
        return messages

    first = matches[0]
    first_user = raw_prompt[: first.start()].strip()
    if first_user:
        messages.append({"role": "user", "content": first_user})

    for idx, match in enumerate(matches):
        role = match.group(1).lower()
        start = match.end()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(raw_prompt)
        content = raw_prompt[start:end].strip()
        if content:
            messages.append({"role": role, "content": content})
    return messages


def build_prompt_text(tokenizer: Any, prompt: str, system_message: str | None = None) -> str:
    messages = parse_helpsteer2_dialogue(prompt, system_message=system_message)
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template is not None:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    rendered = []
    for m in messages:
        rendered.append(f"{m['role'].capitalize()}: {m['content']}")
    rendered.append("Assistant:")
    return "\n".join(rendered)


def tokenize_prompt_response(
    tokenizer: Any,
    prompt_text: str,
    response: str,
    *,
    max_length: int,
    max_prompt_length: int,
    max_response_length: int,
) -> dict[str, list[int]]:
    eos = tokenizer.eos_token or ""
    prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    response_ids = tokenizer(response + eos, add_special_tokens=False)["input_ids"]

    if max_prompt_length is not None and len(prompt_ids) > max_prompt_length:
        prompt_ids = prompt_ids[-max_prompt_length:]
    if max_response_length is not None and len(response_ids) > max_response_length:
        response_ids = response_ids[:max_response_length]

    total = len(prompt_ids) + len(response_ids)
    if total > max_length:
        response_keep = min(len(response_ids), max_response_length, max_length)
        prompt_keep = max(max_length - response_keep, 0)
        response_ids = response_ids[:response_keep]
        prompt_ids = prompt_ids[-prompt_keep:] if prompt_keep > 0 else []

    input_ids = prompt_ids + response_ids
    labels = [-100] * len(prompt_ids) + response_ids
    attention_mask = [1] * len(input_ids)
    return {"input_ids": input_ids, "labels": labels, "attention_mask": attention_mask}


def pad_sequences(
    seqs: list[list[int]],
    pad_value: int,
    *,
    pad_to_multiple_of: int | None = None,
) -> torch.Tensor:
    max_len = max(len(s) for s in seqs)
    if pad_to_multiple_of:
        remainder = max_len % pad_to_multiple_of
        if remainder:
            max_len += pad_to_multiple_of - remainder
    out = torch.full((len(seqs), max_len), int(pad_value), dtype=torch.long)
    for i, seq in enumerate(seqs):
        out[i, : len(seq)] = torch.tensor(seq, dtype=torch.long)
    return out


class PreferencePairDataset(Dataset):
    """JSONL dataset for one objective's z=(x, y_a, y_b, b_i) examples.

    The preferred side is represented by ``preference_label``:
    - 1 means response_a is preferred to response_b;
    - 0 means response_b is preferred to response_a.

    Older JSONL rows with ``chosen``/``rejected`` are accepted and interpreted as
    ``response_a=chosen``, ``response_b=rejected``, ``preference_label=1``.
    """

    def __init__(self, path: str | Path, objective: str | None = None):
        self.path = Path(path)
        self.rows = read_jsonl(self.path)
        if len(self.rows) == 0:
            raise ValueError(f"No rows found in {self.path}")
        self.objective = objective or self.rows[0].get("objective")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.rows[idx]
        response_a = row.get("response_a", row.get("chosen"))
        response_b = row.get("response_b", row.get("rejected"))
        if response_a is None or response_b is None:
            raise KeyError("Each pair row must contain response_a/response_b or chosen/rejected fields.")
        label = int(row.get("preference_label", 1))
        if label not in {0, 1}:
            raise ValueError("preference_label must be 0 or 1.")
        return {
            "prompt": row["prompt"],
            "response_a": response_a,
            "response_b": response_b,
            "preference_label": label,
            "objective": row.get("objective", self.objective),
            "metadata": row,
        }


class DPODataCollator:
    def __init__(
        self,
        tokenizer: Any,
        *,
        max_length: int,
        max_prompt_length: int,
        max_response_length: int,
        system_message: str | None = None,
        pad_to_multiple_of: int | None = 8,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.max_prompt_length = max_prompt_length
        self.max_response_length = max_response_length
        self.system_message = system_message
        self.pad_to_multiple_of = pad_to_multiple_of
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def _encode_side(self, prompt: str, response: str) -> dict[str, list[int]]:
        prompt_text = build_prompt_text(self.tokenizer, prompt, self.system_message)
        return tokenize_prompt_response(
            self.tokenizer,
            prompt_text,
            response,
            max_length=self.max_length,
            max_prompt_length=self.max_prompt_length,
            max_response_length=self.max_response_length,
        )

    def __call__(self, examples: list[dict[str, Any]]) -> dict[str, Any]:
        response_a = [self._encode_side(ex["prompt"], ex["response_a"]) for ex in examples]
        response_b = [self._encode_side(ex["prompt"], ex["response_b"]) for ex in examples]
        pad_id = int(self.tokenizer.pad_token_id)
        batch: dict[str, Any] = {
            "response_a_input_ids": pad_sequences(
                [x["input_ids"] for x in response_a], pad_id, pad_to_multiple_of=self.pad_to_multiple_of
            ),
            "response_a_attention_mask": pad_sequences(
                [x["attention_mask"] for x in response_a], 0, pad_to_multiple_of=self.pad_to_multiple_of
            ),
            "response_a_labels": pad_sequences(
                [x["labels"] for x in response_a], -100, pad_to_multiple_of=self.pad_to_multiple_of
            ),
            "response_b_input_ids": pad_sequences(
                [x["input_ids"] for x in response_b], pad_id, pad_to_multiple_of=self.pad_to_multiple_of
            ),
            "response_b_attention_mask": pad_sequences(
                [x["attention_mask"] for x in response_b], 0, pad_to_multiple_of=self.pad_to_multiple_of
            ),
            "response_b_labels": pad_sequences(
                [x["labels"] for x in response_b], -100, pad_to_multiple_of=self.pad_to_multiple_of
            ),
            "preference_label": torch.tensor([ex["preference_label"] for ex in examples], dtype=torch.float32),
            "objective": [ex["objective"] for ex in examples],
            "metadata": [ex.get("metadata", {}) for ex in examples],
        }
        return batch


def pair_file(pair_dir: str | Path, split: str, objective: str) -> Path:
    return Path(pair_dir) / split / f"{objective}.jsonl"


def build_objective_datasets(
    pair_dir: str | Path,
    split: str,
    objectives: Iterable[str],
) -> dict[str, PreferencePairDataset]:
    datasets: dict[str, PreferencePairDataset] = {}
    for obj in objectives:
        path = pair_file(pair_dir, split, obj)
        if not path.exists():
            raise FileNotFoundError(f"Missing pair file for objective {obj}: {path}")
        datasets[obj] = PreferencePairDataset(path, objective=obj)
    return datasets


def resolve_pair_dir(data_cfg: dict[str, Any], *, eval_mode: bool = False) -> str | Path:
    """Return the configured pair directory for train or eval loading."""
    if eval_mode and data_cfg.get("eval_pair_dir"):
        return data_cfg["eval_pair_dir"]
    if "pair_dir" not in data_cfg:
        raise KeyError("data.pair_dir must be set in the config.")
    return data_cfg["pair_dir"]


def build_attribute_pairs(
    rows: list[dict[str, Any]],
    *,
    split: str,
    objectives: list[str],
    min_score_gap: int = 1,
    noise_rates: dict[str, float] | None = None,
    objective_directions: dict[str, int] | None = None,
    seed: int = 0,
) -> tuple[dict[str, list[dict[str, Any]]], list[PairBuildStats]]:
    """Build objective-specific z=(x, y_a, y_b, b_i) examples from HelpSteer2.

    For each prompt group and objective, two responses become response_a and
    response_b. The preference label b_i is computed from the objective's
    attribute scores, optionally flipped for label-noise experiments.
    """
    noise_rates = noise_rates or {}
    objective_directions = objective_directions or {}
    rng = random.Random(seed)
    groups: dict[str, list[tuple[int, dict[str, Any]]]] = defaultdict(list)
    for idx, row in enumerate(rows):
        if "prompt" not in row or "response" not in row:
            raise KeyError("Expected HelpSteer2 rows with 'prompt' and 'response' fields.")
        groups[str(row["prompt"])].append((idx, row))

    all_pairs: dict[str, list[dict[str, Any]]] = {obj: [] for obj in objectives}
    stats: list[PairBuildStats] = []

    for obj in objectives:
        skipped = 0
        flipped = 0
        direction = int(objective_directions.get(obj, 1))
        if direction not in {-1, 1}:
            raise ValueError(f"Direction for {obj} must be +1 or -1.")
        for prompt, group in groups.items():
            if len(group) < 2:
                continue
            for (idx_a, row_a), (idx_b, row_b) in itertools.combinations(group, 2):
                if obj not in row_a or obj not in row_b:
                    raise KeyError(f"Objective '{obj}' not found in HelpSteer2 row.")
                score_a_raw = int(row_a[obj])
                score_b_raw = int(row_b[obj])
                score_a = direction * score_a_raw
                score_b = direction * score_b_raw
                gap = score_a - score_b
                if abs(gap) < min_score_gap:
                    skipped += 1
                    continue

                preference_label = 1 if gap > 0 else 0
                label_flipped = False
                if rng.random() < float(noise_rates.get(obj, 0.0)):
                    preference_label = 1 - preference_label
                    label_flipped = True
                    flipped += 1

                preferred_score = score_a_raw if preference_label == 1 else score_b_raw
                dispreferred_score = score_b_raw if preference_label == 1 else score_a_raw
                all_pairs[obj].append(
                    {
                        "split": split,
                        "objective": obj,
                        "prompt": prompt,
                        "response_a": str(row_a["response"]),
                        "response_b": str(row_b["response"]),
                        "preference_label": int(preference_label),
                        "response_a_score": int(score_a_raw),
                        "response_b_score": int(score_b_raw),
                        "preferred_score": int(preferred_score),
                        "dispreferred_score": int(dispreferred_score),
                        "score_gap": int(abs(score_a_raw - score_b_raw)),
                        "direction": direction,
                        "source_idx_a": int(idx_a),
                        "source_idx_b": int(idx_b),
                        "label_flipped": label_flipped,
                    }
                )
        stats.append(
            PairBuildStats(
                split=split,
                objective=obj,
                total_pairs=len(all_pairs[obj]),
                skipped_equal_score=skipped,
                flipped_by_noise=flipped,
            )
        )
    return all_pairs, stats
