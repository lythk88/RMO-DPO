#!/usr/bin/env python
from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate every checkpoint in an adapter directory and optionally plot the results."
    )
    parser.add_argument(
        "--repo_root",
        default=str(Path(__file__).resolve().parents[1]),
        help="Repository root that contains scripts/evaluate_helpsteer2.py.",
    )
    parser.add_argument(
        "--python_bin",
        default=".venv/bin/python",
        help="Python executable to use, relative to repo_root unless absolute.",
    )
    parser.add_argument("--config", required=True, help="Evaluation config path.")
    parser.add_argument("--checkpoint_root", required=True, help="Directory containing checkpoint-* adapters.")
    parser.add_argument("--output_dir", required=True, help="Directory for checkpoint-*.json outputs.")
    parser.add_argument(
        "--checkpoint_steps",
        nargs="+",
        type=int,
        default=None,
        help="Optional explicit checkpoint step list to evaluate.",
    )
    parser.add_argument("--split", default="validation", help="Dataset split to evaluate.")
    parser.add_argument(
        "--max_batches",
        type=int,
        default=None,
        help="Optional cap on batches per objective for fast checkpoint snapshots.",
    )
    parser.add_argument(
        "--max_examples",
        type=int,
        default=None,
        help="Optional cap on evaluated examples per objective.",
    )
    parser.add_argument(
        "--per_objective_batch_size",
        type=int,
        default=None,
        help="Optional evaluation batch-size override.",
    )
    parser.add_argument(
        "--skip_existing",
        action="store_true",
        help="Skip checkpoint JSONs that already exist.",
    )
    parser.add_argument(
        "--skip_final",
        action="store_true",
        help="Do not evaluate the checkpoint root itself as final.json.",
    )
    parser.add_argument(
        "--skip_plot",
        action="store_true",
        help="Do not create the standalone checkpoint summary plot.",
    )
    parser.add_argument(
        "--plot_title",
        default="Checkpoint evaluation",
        help="SVG title for the standalone checkpoint summary plot.",
    )
    return parser.parse_args()


def sorted_checkpoint_dirs(checkpoint_root: Path, allowed_steps: set[int] | None = None) -> list[Path]:
    checkpoints = [path for path in checkpoint_root.glob("checkpoint-*") if path.is_dir()]
    if allowed_steps is not None:
        checkpoints = [path for path in checkpoints if int(path.name.split("-", 1)[1]) in allowed_steps]
    return sorted(checkpoints, key=lambda path: int(path.name.split("-", 1)[1]))


def output_name(checkpoint_dir: Path, checkpoint_root: Path) -> str:
    if checkpoint_dir == checkpoint_root:
        return "final.json"
    return f"{checkpoint_dir.name}.json"


def render_path(path: Path, repo_root: Path) -> str:
    try:
        return str(path.resolve().relative_to(repo_root))
    except ValueError:
        return str(path.resolve())


def resolve_python_bin(python_bin: str, repo_root: Path) -> str:
    path = Path(python_bin)
    if path.is_absolute():
        return str(path)
    return str(repo_root / path)


def run_command(cmd: list[str], cwd: Path) -> None:
    print("Running:", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=cwd, check=True)


def main() -> None:
    args = parse_args()
    repo_root = Path(args.repo_root).resolve()
    checkpoint_root = Path(args.checkpoint_root).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = (repo_root / config_path).resolve()
    python_bin = resolve_python_bin(args.python_bin, repo_root)

    allowed_steps = set(args.checkpoint_steps) if args.checkpoint_steps is not None else None
    checkpoints = sorted_checkpoint_dirs(checkpoint_root, allowed_steps=allowed_steps)
    if not args.skip_final and (checkpoint_root / "adapter_config.json").exists():
        checkpoints.append(checkpoint_root)

    for checkpoint_dir in checkpoints:
        output_json = output_dir / output_name(checkpoint_dir, checkpoint_root)
        if args.skip_existing and output_json.exists():
            print(f"Skipping existing {output_json}", flush=True)
            continue
        cmd = [
            python_bin,
            "scripts/evaluate_helpsteer2.py",
            "--config",
            render_path(config_path, repo_root),
            "--checkpoint",
            render_path(checkpoint_dir, repo_root),
            "--split",
            args.split,
            "--output_json",
            render_path(output_json, repo_root),
        ]
        if args.max_batches is not None:
            cmd.extend(["--max_batches", str(args.max_batches)])
        if args.max_examples is not None:
            cmd.extend(["--max_examples", str(args.max_examples)])
        if args.per_objective_batch_size is not None:
            cmd.extend(["--per_objective_batch_size", str(args.per_objective_batch_size)])
        run_command(cmd, repo_root)

    if not args.skip_plot:
        plot_cmd = [
            python_bin,
            "scripts/plot_checkpoint_evals.py",
            "--eval_dir",
            render_path(output_dir, repo_root),
            "--title",
            args.plot_title,
        ]
        run_command(plot_cmd, repo_root)


if __name__ == "__main__":
    main()
