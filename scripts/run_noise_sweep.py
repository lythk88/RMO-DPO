#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any

from rmo_dpo.config import deep_update, load_config

DEFAULT_NOISE_RATES = [0.0, 0.1, 0.2, 0.3]
OBJECTIVES = ["helpfulness", "correctness", "coherence", "complexity", "verbosity"]


def noise_tag(noise_rate: float) -> str:
    return f"{noise_rate:.1f}"


def pair_dir_name(noise_rate: float) -> str:
    return "helpsteer2_pairs" if noise_rate == 0.0 else f"helpsteer2_pairs_noisy_{noise_tag(noise_rate)}"


def build_noise_arg(noise_rate: float) -> str:
    return ",".join(f"{objective}={noise_rate:.1f}" for objective in OBJECTIVES)


def build_config(
    base_cfg: dict[str, Any],
    noise_rate: float,
    *,
    output_root: str = "outputs/noise_sweep",
    run_name_prefix: str = "rmo-dpo-helpsteer2-noise",
    conflict: str | None = None,
    divergence: str | None = None,
) -> dict[str, Any]:
    tag = noise_tag(noise_rate)
    overrides: dict[str, Any] = {
        "output_dir": f"{output_root}/noise_{tag}",
        "run_name": f"{run_name_prefix}-{tag}",
        "data": {
            "pair_dir": f"data/{pair_dir_name(noise_rate)}",
            "eval_pair_dir": "data/helpsteer2_pairs",
        },
    }
    rmo_overrides: dict[str, Any] = {}
    if conflict is not None:
        rmo_overrides["conflict"] = conflict
    if divergence is not None:
        rmo_overrides["divergence"] = divergence
    if rmo_overrides:
        overrides["rmo_dpo"] = rmo_overrides
    return deep_update(
        base_cfg,
        overrides,
    )


def run_command(cmd: list[str], cwd: Path) -> None:
    subprocess.run(cmd, cwd=cwd, check=True)


def ensure_dataset(repo_root: Path, noise_rate: float, seed: int, base_prepare_args: list[str]) -> Path:
    output_dir = repo_root / "data" / pair_dir_name(noise_rate)
    manifest_path = output_dir / "manifest.json"
    if manifest_path.exists():
        return output_dir

    cmd = [
        ".venv/bin/python",
        "scripts/prepare_helpsteer2.py",
        "--output_dir",
        str(output_dir.relative_to(repo_root)),
        "--seed",
        str(seed),
    ]
    if noise_rate > 0.0:
        cmd.extend(["--noise", build_noise_arg(noise_rate)])
    cmd.extend(base_prepare_args)
    run_command(cmd, repo_root)
    return output_dir


def write_config(
    repo_root: Path,
    base_cfg: dict[str, Any],
    noise_rate: float,
    *,
    output_root: str,
    config_dir: str,
    run_name_prefix: str,
    conflict: str | None,
    divergence: str | None,
) -> Path:
    import yaml

    cfg = build_config(
        base_cfg,
        noise_rate,
        output_root=output_root,
        run_name_prefix=run_name_prefix,
        conflict=conflict,
        divergence=divergence,
    )
    config_root = repo_root / config_dir
    config_root.mkdir(parents=True, exist_ok=True)
    config_path = config_root / f"helpsteer2_rmo_dpo_noise_{noise_tag(noise_rate)}.yaml"
    with config_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    return config_path


def evaluate_checkpoint(repo_root: Path, config_path: Path, checkpoint_path: Path, metrics_path: Path) -> None:
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        ".venv/bin/python",
        "scripts/evaluate_helpsteer2.py",
        "--config",
        str(config_path.relative_to(repo_root)),
        "--checkpoint",
        str(checkpoint_path.relative_to(repo_root)),
        "--output_json",
        str(metrics_path.relative_to(repo_root)),
    ]
    run_command(cmd, repo_root)


def train_config(repo_root: Path, config_path: Path) -> None:
    cmd = [
        "bash",
        "scripts/train_rmo_dpo_gpu.sh",
        str(config_path.relative_to(repo_root)),
    ]
    run_command(cmd, repo_root)


def load_metrics(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _polyline_points(xs: list[float], ys: list[float], left: float, top: float, width: float, height: float) -> str:
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    if x_max == x_min:
        x_max = x_min + 1.0
    if y_max == y_min:
        y_max = y_min + 1.0
    points: list[str] = []
    for x, y in zip(xs, ys, strict=True):
        px = left + ((x - x_min) / (x_max - x_min)) * width
        py = top + height - ((y - y_min) / (y_max - y_min)) * height
        points.append(f"{px:.2f},{py:.2f}")
    return " ".join(points)


def _svg_panel(
    title: str,
    xs: list[float],
    series: list[tuple[str, list[float], str]],
    *,
    left: float,
    top: float,
    width: float,
    height: float,
    y_label: str,
) -> str:
    all_y = [value for _, values, _ in series for value in values]
    y_min = min(all_y)
    y_max = max(all_y)
    if y_max == y_min:
        y_max = y_min + 1.0

    parts = [
        f'<rect x="{left}" y="{top}" width="{width}" height="{height}" fill="white" stroke="#333" />',
        f'<text x="{left + width / 2:.2f}" y="{top - 12:.2f}" text-anchor="middle" font-size="16">{title}</text>',
        f'<text x="{left - 38:.2f}" y="{top + height / 2:.2f}" text-anchor="middle" font-size="12" transform="rotate(-90 {left - 38:.2f} {top + height / 2:.2f})">{y_label}</text>',
        f'<text x="{left + width / 2:.2f}" y="{top + height + 34:.2f}" text-anchor="middle" font-size="12">Noise rate</text>',
    ]

    for x in xs:
        px = left if max(xs) == min(xs) else left + ((x - min(xs)) / (max(xs) - min(xs))) * width
        parts.append(f'<line x1="{px:.2f}" y1="{top}" x2="{px:.2f}" y2="{top + height}" stroke="#ddd" />')
        parts.append(f'<text x="{px:.2f}" y="{top + height + 18:.2f}" text-anchor="middle" font-size="11">{x:.1f}</text>')

    for ratio in [0.0, 0.25, 0.5, 0.75, 1.0]:
        py = top + height - ratio * height
        y_value = y_min + ratio * (y_max - y_min)
        parts.append(f'<line x1="{left}" y1="{py:.2f}" x2="{left + width}" y2="{py:.2f}" stroke="#eee" />')
        parts.append(f'<text x="{left - 8:.2f}" y="{py + 4:.2f}" text-anchor="end" font-size="11">{y_value:.3f}</text>')

    legend_x = left + 8
    legend_y = top + 18
    for idx, (label, ys, color) in enumerate(series):
        y_offset = legend_y + idx * 18
        points = _polyline_points(xs, ys, left, top, width, height)
        parts.append(f'<polyline fill="none" stroke="{color}" stroke-width="2.5" points="{points}" />')
        for point in points.split():
            px, py = point.split(",", 1)
            parts.append(f'<circle cx="{px}" cy="{py}" r="3.5" fill="{color}" />')
        parts.append(f'<line x1="{legend_x:.2f}" y1="{y_offset:.2f}" x2="{legend_x + 18:.2f}" y2="{y_offset:.2f}" stroke="{color}" stroke-width="2.5" />')
        parts.append(f'<text x="{legend_x + 24:.2f}" y="{y_offset + 4:.2f}" font-size="11">{label}</text>')
    return "\n".join(parts)


def plot_results(summary: list[dict[str, Any]], output_path: Path) -> None:
    noise_rates = [row["noise_rate"] for row in summary]
    mean_acc = [row["metrics"]["mean_accuracy"] for row in summary]
    worst_acc = [row["metrics"]["worst_accuracy"] for row in summary]
    mean_loss = [row["metrics"]["mean_loss"] for row in summary]
    worst_loss = [row["metrics"]["worst_loss"] for row in summary]
    width = 1200
    height = 460
    margin_left = 70
    panel_width = 470
    panel_height = 320
    top = 60
    gap = 90

    left_panel = _svg_panel(
        "Accuracy vs noise rate",
        noise_rates,
        [("Mean accuracy", mean_acc, "#1f77b4"), ("Worst accuracy", worst_acc, "#d62728")],
        left=margin_left,
        top=top,
        width=panel_width,
        height=panel_height,
        y_label="Accuracy",
    )
    right_panel = _svg_panel(
        "Loss vs noise rate",
        noise_rates,
        [("Mean loss", mean_loss, "#2ca02c"), ("Worst loss", worst_loss, "#ff7f0e")],
        left=margin_left + panel_width + gap,
        top=top,
        width=panel_width,
        height=panel_height,
        y_label="Loss",
    )
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<rect width="100%" height="100%" fill="#f7f7f7" />
<text x="{width / 2:.2f}" y="28" text-anchor="middle" font-size="20">RMO-DPO performance across noise rates</text>
{left_panel}
{right_panel}
</svg>
"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(svg, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare, train, evaluate, and plot an RMO-DPO noise-rate sweep.")
    parser.add_argument("--base_config", default="configs/helpsteer2_rmo_dpo.yaml")
    parser.add_argument("--noise_rates", nargs="+", type=float, default=DEFAULT_NOISE_RATES)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--prepare_only", action="store_true")
    parser.add_argument("--skip_prepare", action="store_true")
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_eval", action="store_true")
    parser.add_argument("--output_root", default="outputs/noise_sweep")
    parser.add_argument("--config_dir", default="configs/noise_sweep")
    parser.add_argument("--run_name_prefix", default="rmo-dpo-helpsteer2-noise")
    parser.add_argument("--conflict", choices=["mgda", "clip", "weighted"])
    parser.add_argument("--divergence", choices=["none", "kl", "chi2"])
    parser.add_argument("--summary_json", default="outputs/noise_sweep/summary.json")
    parser.add_argument("--summary_csv", default="outputs/noise_sweep/summary.csv")
    parser.add_argument("--plot_path", default="outputs/noise_sweep/noise_sweep_metrics.svg")
    parser.add_argument(
        "--prepare_arg",
        action="append",
        default=[],
        help="Extra argument to forward to prepare_helpsteer2.py. Repeat for multiple args.",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    base_cfg = load_config(repo_root / args.base_config)

    summary: list[dict[str, Any]] = []
    for noise_rate in args.noise_rates:
        dataset_dir = repo_root / "data" / pair_dir_name(noise_rate)
        if not args.skip_prepare:
            ensure_dataset(repo_root, noise_rate, args.seed, args.prepare_arg)
        config_path = write_config(
            repo_root,
            base_cfg,
            noise_rate,
            output_root=args.output_root,
            config_dir=args.config_dir,
            run_name_prefix=args.run_name_prefix,
            conflict=args.conflict,
            divergence=args.divergence,
        )
        output_dir = repo_root / build_config(
            base_cfg,
            noise_rate,
            output_root=args.output_root,
            run_name_prefix=args.run_name_prefix,
            conflict=args.conflict,
            divergence=args.divergence,
        )["output_dir"]
        checkpoint_path = output_dir / "final"
        metrics_path = output_dir / "eval_metrics.json"

        if args.prepare_only:
            continue
        if not args.skip_train and not checkpoint_path.exists():
            train_config(repo_root, config_path)
        if not args.skip_eval:
            if not checkpoint_path.exists():
                raise FileNotFoundError(f"Missing checkpoint for noise {noise_tag(noise_rate)}: {checkpoint_path}")
            evaluate_checkpoint(repo_root, config_path, checkpoint_path, metrics_path)

        if metrics_path.exists():
            payload = load_metrics(metrics_path)
            summary.append(
                {
                    "noise_rate": noise_rate,
                    "dataset_dir": str(dataset_dir.relative_to(repo_root)),
                    "config_path": str(config_path.relative_to(repo_root)),
                    "checkpoint_path": str(checkpoint_path.relative_to(repo_root)),
                    "metrics_path": str(metrics_path.relative_to(repo_root)),
                    "metrics": payload["metrics"],
                }
            )

    if args.prepare_only:
        return

    if summary:
        summary_json = repo_root / args.summary_json
        summary_json.parent.mkdir(parents=True, exist_ok=True)
        with summary_json.open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        summary_csv = repo_root / args.summary_csv
        summary_csv.parent.mkdir(parents=True, exist_ok=True)
        metric_keys = [
            "mean_accuracy",
            "worst_accuracy",
            "mean_loss",
            "worst_loss",
        ]
        with summary_csv.open("w", encoding="utf-8") as f:
            f.write("noise_rate," + ",".join(metric_keys) + "\n")
            for row in summary:
                metrics = row["metrics"]
                values = [str(metrics.get(key, "")) for key in metric_keys]
                f.write(f"{row['noise_rate']}," + ",".join(values) + "\n")

        plot_results(summary, repo_root / args.plot_path)


if __name__ == "__main__":
    main()
