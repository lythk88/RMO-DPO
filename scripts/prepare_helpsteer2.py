#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from datasets import load_dataset

from rmo_dpo.data import HELPSTEER2_OBJECTIVES, build_attribute_pairs, noise_rates_for_split
from rmo_dpo.utils import seed_everything, setup_logger, write_jsonl


def parse_noise(spec: str | None) -> dict[str, float]:
    if not spec:
        return {}
    out: dict[str, float] = {}
    for item in spec.split(","):
        if not item.strip():
            continue
        key, value = item.split("=", 1)
        out[key.strip()] = float(value)
    return out


def parse_directions(spec: str | None) -> dict[str, int]:
    if not spec:
        return {}
    out: dict[str, int] = {}
    for item in spec.split(","):
        if not item.strip():
            continue
        key, value = item.split("=", 1)
        v = value.strip()
        if v in {"+", "+1", "1", "higher"}:
            out[key.strip()] = 1
        elif v in {"-", "-1", "lower"}:
            out[key.strip()] = -1
        else:
            raise ValueError(f"Invalid direction {item!r}; use +1 or -1.")
    return out


def maybe_limit(rows: list[dict[str, Any]], limit: int | None) -> list[dict[str, Any]]:
    if limit is None or limit <= 0:
        return rows
    return rows[:limit]


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare HelpSteer2 objective-specific pairs.")
    parser.add_argument("--output_dir", default="data/helpsteer2_pairs")
    parser.add_argument("--dataset_name", default="nvidia/HelpSteer2")
    parser.add_argument("--objectives", nargs="+", default=HELPSTEER2_OBJECTIVES)
    parser.add_argument("--min_score_gap", type=int, default=1)
    parser.add_argument("--noise", default=None, help="Comma list, e.g. helpfulness=0.05,correctness=0.30")
    parser.add_argument(
        "--directions",
        default=None,
        help="Comma list of objective directions, e.g. verbosity=-1. Default: all +1.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--train_limit", type=int, default=None)
    parser.add_argument("--validation_limit", type=int, default=None)
    args = parser.parse_args()

    logger = setup_logger("prepare_helpsteer2")
    seed_everything(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading %s", args.dataset_name)
    ds = load_dataset(args.dataset_name)
    noise = parse_noise(args.noise)
    directions = parse_directions(args.directions)
    manifest: dict[str, Any] = {
        "dataset_name": args.dataset_name,
        "objectives": args.objectives,
        "min_score_gap": args.min_score_gap,
        "noise": noise,
        "directions": directions,
        "splits": {},
    }

    for split, limit in [("train", args.train_limit), ("validation", args.validation_limit)]:
        if split not in ds:
            logger.warning("Split %s not found; skipping", split)
            continue
        rows = [dict(x) for x in ds[split]]
        rows = maybe_limit(rows, limit)
        split_noise = noise_rates_for_split(split, noise)
        pairs, stats = build_attribute_pairs(
            rows,
            split=split,
            objectives=args.objectives,
            min_score_gap=args.min_score_gap,
            noise_rates=split_noise,
            objective_directions=directions,
            seed=args.seed + (0 if split == "train" else 10_000),
        )
        manifest["splits"][split] = {"noise": split_noise}
        for stat in stats:
            path = output_dir / split / f"{stat.objective}.jsonl"
            n = write_jsonl(path, pairs[stat.objective])
            manifest["splits"][split][stat.objective] = {
                "path": str(path),
                "total_pairs": stat.total_pairs,
                "skipped_equal_score": stat.skipped_equal_score,
                "flipped_by_noise": stat.flipped_by_noise,
            }
            logger.info(
                "%s/%s: wrote %d pairs (skipped=%d, noisy_flips=%d)",
                split,
                stat.objective,
                n,
                stat.skipped_equal_score,
                stat.flipped_by_noise,
            )

    with (output_dir / "manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    logger.info("Wrote manifest to %s", output_dir / "manifest.json")


if __name__ == "__main__":
    main()
