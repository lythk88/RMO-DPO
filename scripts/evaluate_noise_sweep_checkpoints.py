#!/usr/bin/env python
from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate every checkpoint in a noise sweep against the corresponding validation split."
    )
    parser.add_argument(
        "--repo_root",
        default=str(Path(__file__).resolve().parents[1]),
        help="Repository root.",
    )
    parser.add_argument(
        "--noise_rates",
        nargs="+",
        type=float,
        default=[0.1, 0.2, 0.3],
        help="Noise rates to evaluate.",
    )
    parser.add_argument(
        "--eval_dir_name",
        default="evals",
        help="Per-noise output subdirectory for checkpoint JSON files.",
    )
    parser.add_argument(
        "--split",
        default="validation",
        help="Dataset split to evaluate. The noise-sweep configs use validation as the test-like holdout split.",
    )
    parser.add_argument(
        "--max_batches",
        type=int,
        default=None,
        help="Optional cap on batches per objective for faster probes.",
    )
    parser.add_argument(
        "--skip_existing",
        action="store_true",
        help="Skip checkpoint JSON files that already exist.",
    )
    parser.add_argument(
        "--skip_final",
        action="store_true",
        help="Do not evaluate the final adapter directory.",
    )
    parser.add_argument(
        "--skip_plot",
        action="store_true",
        help="Do not run the post-hoc plotting step after evaluations.",
    )
    parser.add_argument(
        "--plot_output_dir",
        default="outputs/noise_sweep/posthoc_analysis",
        help="Plot output directory relative to repo_root.",
    )
    return parser.parse_args()


def noise_tag(noise_rate: float) -> str:
    return f"{noise_rate:.1f}"


def sorted_checkpoint_dirs(noise_dir: Path) -> list[Path]:
    checkpoints = [path for path in noise_dir.glob("checkpoint-*") if path.is_dir()]
    return sorted(checkpoints, key=lambda path: int(path.name.split("-", 1)[1]))


def run_command(cmd: list[str], cwd: Path) -> None:
    print("Running:", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=cwd, check=True)


def checkpoint_output_name(checkpoint_dir: Path) -> str:
    return f"{checkpoint_dir.name}.json" if checkpoint_dir.name.startswith("checkpoint-") else "final.json"


def main() -> None:
    args = parse_args()
    repo_root = Path(args.repo_root).resolve()

    for noise_rate in args.noise_rates:
        tag = noise_tag(noise_rate)
        noise_dir = repo_root / "outputs" / "noise_sweep" / f"noise_{tag}"
        config_path = repo_root / "configs" / "noise_sweep" / f"helpsteer2_rmo_dpo_noise_{tag}.yaml"
        eval_dir = noise_dir / args.eval_dir_name
        eval_dir.mkdir(parents=True, exist_ok=True)

        checkpoints = sorted_checkpoint_dirs(noise_dir)
        if not args.skip_final:
            final_dir = noise_dir / "final"
            if final_dir.exists():
                checkpoints.append(final_dir)

        for checkpoint_dir in checkpoints:
            output_json = eval_dir / checkpoint_output_name(checkpoint_dir)
            if args.skip_existing and output_json.exists():
                print(f"Skipping existing {output_json}", flush=True)
                continue
            cmd = [
                ".venv/bin/python",
                "scripts/evaluate_helpsteer2.py",
                "--config",
                str(config_path.relative_to(repo_root)),
                "--checkpoint",
                str(checkpoint_dir.relative_to(repo_root)),
                "--split",
                args.split,
                "--output_json",
                str(output_json.relative_to(repo_root)),
            ]
            if args.max_batches is not None:
                cmd.extend(["--max_batches", str(args.max_batches)])
            run_command(cmd, repo_root)

    if not args.skip_plot:
        plot_cmd = [
            ".venv/bin/python",
            "scripts/plot_noise_sweep_posthoc_eval_analysis.py",
            "--repo_root",
            str(repo_root),
            "--eval_dir_name",
            args.eval_dir_name,
            "--output_dir",
            args.plot_output_dir,
            "--noise_rates",
            *[str(rate) for rate in args.noise_rates],
        ]
        run_command(plot_cmd, repo_root)


if __name__ == "__main__":
    main()
