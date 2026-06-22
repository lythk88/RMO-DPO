#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from datasets import load_dataset

from rmo_dpo.utils import seed_everything, setup_logger, write_jsonl

OBJECTIVE_TO_FIELD = {
    "helpfulness": "better_response_id",
    "harmlessness": "safer_response_id",
}
SPLIT_RENAMES = {
    "train": "train",
    "test": "validation",
}


def maybe_limit(rows: list[dict[str, Any]], limit: int | None) -> list[dict[str, Any]]:
    if limit is None or limit <= 0:
        return rows
    return rows[:limit]


def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value)


def normalize_preference(value: Any, field_name: str) -> int:
    if value not in {0, 1, 0.0, 1.0, True, False}:
        raise ValueError(f"{field_name} must be 0 or 1; got {value!r}")
    return int(value)


def build_pair_row(
    row: dict[str, Any],
    *,
    split: str,
    objective: str,
    preference_field: str,
    source_idx: int,
) -> dict[str, Any]:
    preferred_response_id = normalize_preference(row.get(preference_field), preference_field)
    prompt = normalize_text(row.get("prompt"))
    response_0 = normalize_text(row.get("response_0"))
    response_1 = normalize_text(row.get("response_1"))
    return {
        "split": split,
        "objective": objective,
        "prompt": prompt,
        "response_a": response_0,
        "response_b": response_1,
        "preference_label": 1 if preferred_response_id == 0 else 0,
        "preferred_response_id": preferred_response_id,
        "preference_field": preference_field,
        # Keep a stable pair key across objectives so RACO can merge the same examples later.
        "source_idx_a": source_idx * 2,
        "source_idx_b": source_idx * 2 + 1,
        "prompt_source": row.get("prompt_source"),
        "response_0_source": row.get("response_0_source"),
        "response_1_source": row.get("response_1_source"),
        "better_response_id": normalize_preference(row.get("better_response_id"), "better_response_id"),
        "safer_response_id": normalize_preference(row.get("safer_response_id"), "safer_response_id"),
        "is_response_0_safe": row.get("is_response_0_safe"),
        "is_response_1_safe": row.get("is_response_1_safe"),
        "response_0_harm_category": row.get("response_0_harm_category"),
        "response_1_harm_category": row.get("response_1_harm_category"),
        "response_0_severity_level": row.get("response_0_severity_level"),
        "response_1_severity_level": row.get("response_1_severity_level"),
        "response_0_sha256": row.get("response_0_sha256"),
        "response_1_sha256": row.get("response_1_sha256"),
    }


def dataset_kwargs(dataset_name: str, dataset_config: str | None, revision: str | None) -> dict[str, Any]:
    kwargs: dict[str, Any] = {"path": dataset_name}
    if dataset_config:
        kwargs["name"] = dataset_config
    if revision:
        kwargs["revision"] = revision
    return kwargs


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare PKU-SafeRLHF objective-specific pairs.")
    parser.add_argument("--output_dir", default="data/pku_saferlhf_pairs")
    parser.add_argument("--dataset_name", default="PKU-Alignment/PKU-SafeRLHF")
    parser.add_argument(
        "--dataset_config",
        default=None,
        help="Optional Hugging Face config/subset name, e.g. alpaca-7b or alpaca3-8b.",
    )
    parser.add_argument("--revision", default=None, help="Optional dataset revision, e.g. v0.")
    parser.add_argument(
        "--objectives",
        nargs="+",
        default=list(OBJECTIVE_TO_FIELD),
        choices=sorted(OBJECTIVE_TO_FIELD),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train_limit", type=int, default=None)
    parser.add_argument("--validation_limit", type=int, default=None)
    args = parser.parse_args()

    logger = setup_logger("prepare_pku_saferlhf")
    seed_everything(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    hf_kwargs = dataset_kwargs(args.dataset_name, args.dataset_config, args.revision)
    logger.info("Loading %s", hf_kwargs)
    ds = load_dataset(**hf_kwargs)

    manifest: dict[str, Any] = {
        "dataset_name": args.dataset_name,
        "dataset_config": args.dataset_config,
        "revision": args.revision,
        "objectives": args.objectives,
        "split_map": SPLIT_RENAMES,
        "splits": {},
    }

    split_limits = {
        "train": args.train_limit,
        "test": args.validation_limit,
    }
    for source_split, target_split in SPLIT_RENAMES.items():
        if source_split not in ds:
            logger.warning("Split %s not found; skipping", source_split)
            continue
        rows = [dict(x) for x in ds[source_split]]
        rows = maybe_limit(rows, split_limits[source_split])
        split_manifest: dict[str, Any] = {
            "source_split": source_split,
            "rows_loaded": len(rows),
        }
        for objective in args.objectives:
            preference_field = OBJECTIVE_TO_FIELD[objective]
            pair_rows = [
                build_pair_row(
                    row,
                    split=target_split,
                    objective=objective,
                    preference_field=preference_field,
                    source_idx=idx,
                )
                for idx, row in enumerate(rows)
            ]
            path = output_dir / target_split / f"{objective}.jsonl"
            n = write_jsonl(path, pair_rows)
            split_manifest[objective] = {
                "path": str(path),
                "rows_written": n,
                "preference_field": preference_field,
            }
            logger.info("%s/%s: wrote %d pairs", target_split, objective, n)
        manifest["splits"][target_split] = split_manifest

    manifest_path = output_dir / "manifest.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    logger.info("Wrote manifest to %s", manifest_path)


if __name__ == "__main__":
    main()
