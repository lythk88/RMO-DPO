#!/usr/bin/env python
from __future__ import annotations

import argparse
import ast
import csv
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


OBJECTIVES = ["helpfulness", "correctness", "coherence", "complexity", "verbosity"]
OBJECTIVE_COLORS = {
    "helpfulness": "#2a9d8f",
    "correctness": "#457b9d",
    "coherence": "#8d5fd3",
    "complexity": "#e76f51",
    "verbosity": "#f4a261",
}
NOISE_COLORS = {
    0.1: "#1d3557",
    0.2: "#e76f51",
    0.3: "#2a9d8f",
}
EVAL_LINE_RE = re.compile(r"eval step=(\d+) (\{.*\})")
NOISE_RE = re.compile(r"noise_(\d+(?:\.\d+)?)")


@dataclass
class NoiseRun:
    noise_rate: float
    output_dir: Path
    config_path: Path
    log_path: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create checkpoint/noise-rate plots from logged training evaluations and final eval JSON files."
    )
    parser.add_argument(
        "--repo_root",
        default=str(Path(__file__).resolve().parents[1]),
        help="Repository root that contains wandb/ and outputs/noise_sweep/.",
    )
    parser.add_argument(
        "--noise_rates",
        nargs="+",
        type=float,
        default=[0.1, 0.2, 0.3],
        help="Noise rates to include in the analysis.",
    )
    parser.add_argument(
        "--output_dir",
        default="outputs/noise_sweep/analysis",
        help="Output directory relative to repo_root.",
    )
    return parser.parse_args()


def noise_tag(noise_rate: float) -> str:
    return f"{noise_rate:.1f}"


def parse_noise_from_output_dir(output_dir: str) -> float:
    match = NOISE_RE.search(output_dir)
    if not match:
        raise ValueError(f"Could not parse noise rate from output dir: {output_dir}")
    return float(match.group(1))


def discover_noise_runs(repo_root: Path, noise_rates: list[float]) -> list[NoiseRun]:
    wanted = {round(rate, 3) for rate in noise_rates}
    chosen: dict[float, tuple[tuple[int, float], NoiseRun]] = {}
    for config_path in repo_root.glob("wandb/run-*/files/config.yaml"):
        try:
            payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        output_dir_value = payload.get("output_dir", {}).get("value")
        if not isinstance(output_dir_value, str):
            continue
        if not output_dir_value.startswith("outputs/noise_sweep/noise_"):
            continue
        noise_rate = round(parse_noise_from_output_dir(output_dir_value), 3)
        if noise_rate not in wanted:
            continue
        log_path = config_path.parent / "output.log"
        if not log_path.exists():
            continue
        run = NoiseRun(
            noise_rate=noise_rate,
            output_dir=repo_root / output_dir_value,
            config_path=config_path,
            log_path=log_path,
        )
        rank = (int(log_path.stat().st_size), float(log_path.stat().st_mtime))
        previous = chosen.get(noise_rate)
        if previous is None or rank > previous[0]:
            chosen[noise_rate] = (rank, run)
    missing = sorted(wanted - set(chosen))
    if missing:
        raise FileNotFoundError(f"Missing wandb training logs for noise rates: {', '.join(f'{x:.1f}' for x in missing)}")
    return [chosen[rate][1] for rate in sorted(chosen)]


def canonicalize_training_metrics(eval_payload: dict[str, Any]) -> dict[str, float]:
    metrics = {
        "mean_accuracy": float(eval_payload["eval/mean_acc"]),
        "worst_accuracy": float(eval_payload["eval/worst_acc"]),
        "mean_loss": float(eval_payload["eval/mean_loss"]),
        "worst_loss": float(eval_payload["eval/worst_loss"]),
    }
    for objective in OBJECTIVES:
        metrics[f"{objective}/accuracy"] = float(eval_payload[f"eval/{objective}_acc"])
        metrics[f"{objective}/loss"] = float(eval_payload[f"eval/{objective}_loss"])
        metrics[f"{objective}/margin"] = float(eval_payload[f"eval/{objective}_margin"])
    return metrics


def parse_training_eval_rows(log_path: Path) -> list[dict[str, Any]]:
    rows_by_step: dict[int, dict[str, Any]] = {}
    for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = EVAL_LINE_RE.search(line)
        if not match:
            continue
        step = int(match.group(1))
        payload = ast.literal_eval(match.group(2))
        rows_by_step[step] = {
            "step": step,
            "metrics": canonicalize_training_metrics(payload),
        }
    if not rows_by_step:
        raise ValueError(f"No checkpoint eval rows found in {log_path}")
    return [rows_by_step[step] for step in sorted(rows_by_step)]


def load_final_eval_metrics(noise_dir: Path) -> dict[str, float]:
    payload = json.loads((noise_dir / "eval_metrics.json").read_text(encoding="utf-8"))
    return {key: float(value) for key, value in payload["metrics"].items()}


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_json(path: Path, payload: Any) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def blend_rgb(color_a: tuple[int, int, int], color_b: tuple[int, int, int], t: float) -> str:
    t = min(1.0, max(0.0, t))
    r = round(color_a[0] + (color_b[0] - color_a[0]) * t)
    g = round(color_a[1] + (color_b[1] - color_a[1]) * t)
    b = round(color_a[2] + (color_b[2] - color_a[2]) * t)
    return f"#{r:02x}{g:02x}{b:02x}"


def svg_header(width: int, height: int, title: str) -> list[str]:
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#faf9f6"/>',
        f'<text x="{width / 2:.1f}" y="28" text-anchor="middle" font-size="22" font-family="sans-serif">{title}</text>',
    ]


def close_svg(parts: list[str], path: Path) -> None:
    parts.append("</svg>")
    ensure_dir(path.parent)
    path.write_text("\n".join(parts), encoding="utf-8")


def polyline(points: list[tuple[float, float]]) -> str:
    return " ".join(f"{x:.2f},{y:.2f}" for x, y in points)


def scale_points(
    xs: list[float],
    ys: list[float],
    x0: float,
    y0: float,
    w: float,
    h: float,
    xmin: float,
    xmax: float,
    ymin: float,
    ymax: float,
) -> list[tuple[float, float]]:
    points: list[tuple[float, float]] = []
    for x_value, y_value in zip(xs, ys, strict=True):
        if xmax > xmin:
            px = x0 + (x_value - xmin) / (xmax - xmin) * w
        else:
            px = x0 + w / 2
        if ymax > ymin:
            py = y0 + h - (y_value - ymin) / (ymax - ymin) * h
        else:
            py = y0 + h / 2
        points.append((px, py))
    return points


def choose_ticks(xs: list[int], limit: int = 6) -> list[int]:
    if len(xs) <= limit:
        return xs
    idxs = {0, len(xs) - 1}
    for i in range(1, limit - 1):
        idx = round(i * (len(xs) - 1) / (limit - 1))
        idxs.add(idx)
    return [xs[idx] for idx in sorted(idxs)]


def line_panel(
    parts: list[str],
    title: str,
    xs: list[int],
    series: list[tuple[str, list[float], str]],
    *,
    x0: float,
    y0: float,
    w: float,
    h: float,
    x_label: str,
    y_label: str,
    value_format: str = "{:.3f}",
    ymin: float | None = None,
    ymax: float | None = None,
) -> None:
    all_values = [value for _, values, _ in series for value in values]
    plot_ymin = min(all_values) if ymin is None else ymin
    plot_ymax = max(all_values) if ymax is None else ymax
    if math.isclose(plot_ymax, plot_ymin):
        plot_ymax = plot_ymin + 1.0
    if ymin is None or ymax is None:
        pad = (plot_ymax - plot_ymin) * 0.08
        plot_ymin -= pad
        plot_ymax += pad

    parts.append(f'<rect x="{x0}" y="{y0}" width="{w}" height="{h}" fill="white" stroke="#d7d7d7"/>')
    parts.append(
        f'<text x="{x0 + w / 2:.1f}" y="{y0 - 12:.1f}" text-anchor="middle" font-size="17" font-family="sans-serif">{title}</text>'
    )
    for frac in [0.0, 0.25, 0.5, 0.75, 1.0]:
        py = y0 + h - frac * h
        y_value = plot_ymin + frac * (plot_ymax - plot_ymin)
        parts.append(f'<line x1="{x0}" y1="{py:.1f}" x2="{x0 + w}" y2="{py:.1f}" stroke="#ededed"/>')
        parts.append(
            f'<text x="{x0 - 8:.1f}" y="{py + 4:.1f}" text-anchor="end" font-size="11" font-family="sans-serif">{value_format.format(y_value)}</text>'
        )

    tick_values = choose_ticks(xs)
    xmin, xmax = min(xs), max(xs)
    for tick in tick_values:
        px = x0 + ((tick - xmin) / (xmax - xmin) * w if xmax > xmin else 0.5 * w)
        parts.append(f'<line x1="{px:.1f}" y1="{y0}" x2="{px:.1f}" y2="{y0 + h}" stroke="#f1f1f1"/>')
        parts.append(
            f'<text x="{px:.1f}" y="{y0 + h + 18:.1f}" text-anchor="middle" font-size="11" font-family="sans-serif">{tick}</text>'
        )
    parts.append(
        f'<text x="{x0 + w / 2:.1f}" y="{y0 + h + 36:.1f}" text-anchor="middle" font-size="12" font-family="sans-serif">{x_label}</text>'
    )
    parts.append(
        f'<text x="{x0 - 44:.1f}" y="{y0 + h / 2:.1f}" text-anchor="middle" font-size="12" font-family="sans-serif" transform="rotate(-90 {x0 - 44:.1f},{y0 + h / 2:.1f})">{y_label}</text>'
    )

    for label, values, color in series:
        points = scale_points(xs, values, x0, y0, w, h, xmin, xmax, plot_ymin, plot_ymax)
        parts.append(f'<polyline fill="none" stroke="{color}" stroke-width="2.5" points="{polyline(points)}"/>')
        for px, py in points:
            parts.append(f'<circle cx="{px:.2f}" cy="{py:.2f}" r="3.1" fill="{color}"/>')

    legend_x = x0 + 10
    legend_y = y0 + 18
    for index, (label, _, color) in enumerate(series):
        ly = legend_y + index * 18
        parts.append(f'<line x1="{legend_x}" y1="{ly}" x2="{legend_x + 18}" y2="{ly}" stroke="{color}" stroke-width="2.5"/>')
        parts.append(f'<text x="{legend_x + 24}" y="{ly + 4}" font-size="11" font-family="sans-serif">{label}</text>')


def heatmap_panel(
    parts: list[str],
    title: str,
    matrix: list[list[float]],
    x_labels: list[int],
    y_labels: list[str],
    *,
    x0: float,
    y0: float,
    w: float,
    h: float,
    value_format: str,
) -> None:
    flat_values = [value for row in matrix for value in row]
    vmin = min(flat_values)
    vmax = max(flat_values)
    if math.isclose(vmin, vmax):
        vmax = vmin + 1.0
    cell_w = w / len(x_labels)
    cell_h = h / len(y_labels)
    parts.append(f'<rect x="{x0}" y="{y0}" width="{w}" height="{h}" fill="white" stroke="#d7d7d7"/>')
    parts.append(
        f'<text x="{x0 + w / 2:.1f}" y="{y0 - 12:.1f}" text-anchor="middle" font-size="17" font-family="sans-serif">{title}</text>'
    )
    for row_index, (noise_label, row) in enumerate(zip(y_labels, matrix, strict=True)):
        py = y0 + row_index * cell_h
        parts.append(
            f'<text x="{x0 - 8:.1f}" y="{py + cell_h / 2 + 4:.1f}" text-anchor="end" font-size="11" font-family="sans-serif">{noise_label}</text>'
        )
        for col_index, value in enumerate(row):
            px = x0 + col_index * cell_w
            shade = (value - vmin) / (vmax - vmin) if vmax > vmin else 0.5
            fill = blend_rgb((245, 239, 230), (29, 83, 138), shade)
            text_fill = "#0f172a" if shade < 0.45 else "white"
            parts.append(
                f'<rect x="{px:.2f}" y="{py:.2f}" width="{cell_w:.2f}" height="{cell_h:.2f}" fill="{fill}" stroke="#f6f6f6"/>'
            )
            parts.append(
                f'<text x="{px + cell_w / 2:.2f}" y="{py + cell_h / 2 + 4:.2f}" text-anchor="middle" font-size="10" font-family="sans-serif" fill="{text_fill}">{value_format.format(value)}</text>'
            )
    tick_values = choose_ticks(x_labels)
    for tick in tick_values:
        idx = x_labels.index(tick)
        px = x0 + (idx + 0.5) * cell_w
        parts.append(
            f'<text x="{px:.1f}" y="{y0 + h + 18:.1f}" text-anchor="middle" font-size="11" font-family="sans-serif">{tick}</text>'
        )
    parts.append(
        f'<text x="{x0 + w / 2:.1f}" y="{y0 + h + 36:.1f}" text-anchor="middle" font-size="12" font-family="sans-serif">Checkpoint step</text>'
    )
    parts.append(
        f'<text x="{x0 - 44:.1f}" y="{y0 + h / 2:.1f}" text-anchor="middle" font-size="12" font-family="sans-serif" transform="rotate(-90 {x0 - 44:.1f},{y0 + h / 2:.1f})">Noise rate</text>'
    )

    legend_x = x0 + w + 12
    legend_y = y0 + 12
    steps = 5
    for idx in range(steps):
        frac = idx / (steps - 1)
        py = legend_y + idx * 24
        fill = blend_rgb((245, 239, 230), (29, 83, 138), frac)
        value = vmin + frac * (vmax - vmin)
        parts.append(f'<rect x="{legend_x}" y="{py}" width="14" height="14" fill="{fill}" stroke="#cccccc"/>')
        parts.append(
            f'<text x="{legend_x + 22}" y="{py + 11}" font-size="10" font-family="sans-serif">{value_format.format(value)}</text>'
        )


def write_per_noise_plot(path: Path, noise_rate: float, rows: list[dict[str, Any]]) -> None:
    xs = [row["step"] for row in rows]
    width, height = 1260, 920
    panel_w, panel_h = 440, 300
    parts = svg_header(
        width,
        height,
        f"Noise {noise_rate:.1f} checkpoint performance from training logs (validation, max_batches=50)",
    )
    aggregate_acc = [
        ("Mean accuracy", [row["metrics"]["mean_accuracy"] for row in rows], "#1d3557"),
        ("Worst accuracy", [row["metrics"]["worst_accuracy"] for row in rows], "#d1495b"),
    ]
    aggregate_loss = [
        ("Mean loss", [row["metrics"]["mean_loss"] for row in rows], "#2a9d8f"),
        ("Worst loss", [row["metrics"]["worst_loss"] for row in rows], "#f4a261"),
    ]
    objective_acc = [
        (objective, [row["metrics"][f"{objective}/accuracy"] for row in rows], OBJECTIVE_COLORS[objective])
        for objective in OBJECTIVES
    ]
    objective_loss = [
        (objective, [row["metrics"][f"{objective}/loss"] for row in rows], OBJECTIVE_COLORS[objective])
        for objective in OBJECTIVES
    ]
    line_panel(parts, "Aggregate Accuracy", xs, aggregate_acc, x0=70, y0=70, w=panel_w, h=panel_h, x_label="Checkpoint step", y_label="Accuracy", ymin=0.30, ymax=0.65)
    line_panel(parts, "Aggregate Loss", xs, aggregate_loss, x0=70, y0=480, w=panel_w, h=panel_h, x_label="Checkpoint step", y_label="Loss")
    line_panel(parts, "Objective Accuracy", xs, objective_acc, x0=680, y0=70, w=panel_w, h=panel_h, x_label="Checkpoint step", y_label="Accuracy", ymin=0.30, ymax=0.80)
    line_panel(parts, "Objective Loss", xs, objective_loss, x0=680, y0=480, w=panel_w, h=panel_h, x_label="Checkpoint step", y_label="Loss")
    close_svg(parts, path)


def write_combined_checkpoint_accuracy_plot(path: Path, series_by_noise: list[tuple[float, list[dict[str, Any]]]]) -> None:
    xs = [row["step"] for row in series_by_noise[0][1]]
    width, height = 1240, 500
    parts = svg_header(width, height, "Checkpoint accuracy across noise rates (training-time validation, max_batches=50)")
    mean_series = []
    worst_series = []
    for noise_rate, rows in series_by_noise:
        label = f"Noise {noise_rate:.1f}"
        color = NOISE_COLORS.get(noise_rate, "#555555")
        mean_series.append((label, [row["metrics"]["mean_accuracy"] for row in rows], color))
        worst_series.append((label, [row["metrics"]["worst_accuracy"] for row in rows], color))
    line_panel(parts, "Mean Accuracy", xs, mean_series, x0=70, y0=70, w=470, h=310, x_label="Checkpoint step", y_label="Accuracy", ymin=0.42, ymax=0.62)
    line_panel(parts, "Worst Accuracy", xs, worst_series, x0=670, y0=70, w=470, h=310, x_label="Checkpoint step", y_label="Accuracy", ymin=0.30, ymax=0.55)
    close_svg(parts, path)


def write_combined_checkpoint_loss_plot(path: Path, series_by_noise: list[tuple[float, list[dict[str, Any]]]]) -> None:
    xs = [row["step"] for row in series_by_noise[0][1]]
    width, height = 1240, 500
    parts = svg_header(width, height, "Checkpoint loss across noise rates (training-time validation, max_batches=50)")
    mean_series = []
    worst_series = []
    for noise_rate, rows in series_by_noise:
        label = f"Noise {noise_rate:.1f}"
        color = NOISE_COLORS.get(noise_rate, "#555555")
        mean_series.append((label, [row["metrics"]["mean_loss"] for row in rows], color))
        worst_series.append((label, [row["metrics"]["worst_loss"] for row in rows], color))
    line_panel(parts, "Mean Loss", xs, mean_series, x0=70, y0=70, w=470, h=310, x_label="Checkpoint step", y_label="Loss")
    line_panel(parts, "Worst Loss", xs, worst_series, x0=670, y0=70, w=470, h=310, x_label="Checkpoint step", y_label="Loss")
    close_svg(parts, path)


def write_heatmap_plot(path: Path, series_by_noise: list[tuple[float, list[dict[str, Any]]]]) -> None:
    xs = [row["step"] for row in series_by_noise[0][1]]
    y_labels = [f"{noise_rate:.1f}" for noise_rate, _ in series_by_noise]

    def metric_matrix(metric_name: str) -> list[list[float]]:
        return [[row["metrics"][metric_name] for row in rows] for _, rows in series_by_noise]

    width, height = 1420, 860
    parts = svg_header(width, height, "Checkpoint metric heatmaps across noise rates")
    heatmap_panel(parts, "Mean Accuracy", metric_matrix("mean_accuracy"), xs, y_labels, x0=90, y0=70, w=500, h=260, value_format="{:.2f}")
    heatmap_panel(parts, "Worst Accuracy", metric_matrix("worst_accuracy"), xs, y_labels, x0=760, y0=70, w=500, h=260, value_format="{:.2f}")
    heatmap_panel(parts, "Mean Loss", metric_matrix("mean_loss"), xs, y_labels, x0=90, y0=470, w=500, h=260, value_format="{:.3f}")
    heatmap_panel(parts, "Worst Loss", metric_matrix("worst_loss"), xs, y_labels, x0=760, y0=470, w=500, h=260, value_format="{:.3f}")
    close_svg(parts, path)


def write_final_eval_plot(path: Path, final_rows: list[dict[str, Any]]) -> None:
    noise_rates = [row["noise_rate"] for row in final_rows]
    width, height = 1240, 500
    parts = svg_header(width, height, "Final full-validation metrics across noise rates")
    accuracy_series = [
        ("Mean accuracy", [row["mean_accuracy"] for row in final_rows], "#1d3557"),
        ("Worst accuracy", [row["worst_accuracy"] for row in final_rows], "#d1495b"),
    ]
    loss_series = [
        ("Mean loss", [row["mean_loss"] for row in final_rows], "#2a9d8f"),
        ("Worst loss", [row["worst_loss"] for row in final_rows], "#f4a261"),
    ]
    line_panel(parts, "Accuracy vs Noise", noise_rates, accuracy_series, x0=70, y0=70, w=470, h=310, x_label="Noise rate", y_label="Accuracy", ymin=0.47, ymax=0.60, value_format="{:.3f}")
    line_panel(parts, "Loss vs Noise", noise_rates, loss_series, x0=670, y0=70, w=470, h=310, x_label="Noise rate", y_label="Loss", value_format="{:.3f}")
    close_svg(parts, path)


def build_training_csv_rows(series_by_noise: list[tuple[float, list[dict[str, Any]]]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for noise_rate, series in series_by_noise:
        for row in series:
            metrics = row["metrics"]
            out_row = {
                "noise_rate": noise_rate,
                "step": row["step"],
                "mean_accuracy": metrics["mean_accuracy"],
                "worst_accuracy": metrics["worst_accuracy"],
                "mean_loss": metrics["mean_loss"],
                "worst_loss": metrics["worst_loss"],
            }
            for objective in OBJECTIVES:
                out_row[f"{objective}_accuracy"] = metrics[f"{objective}/accuracy"]
                out_row[f"{objective}_loss"] = metrics[f"{objective}/loss"]
            rows.append(out_row)
    return rows


def build_final_csv_rows(series_by_noise: list[tuple[float, dict[str, float]]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for noise_rate, metrics in series_by_noise:
        row = {
            "noise_rate": noise_rate,
            "mean_accuracy": metrics["mean_accuracy"],
            "worst_accuracy": metrics["worst_accuracy"],
            "mean_loss": metrics["mean_loss"],
            "worst_loss": metrics["worst_loss"],
        }
        for objective in OBJECTIVES:
            row[f"{objective}_accuracy"] = metrics[f"{objective}/accuracy"]
            row[f"{objective}_loss"] = metrics[f"{objective}/loss"]
        rows.append(row)
    return rows


def best_checkpoint_summary(noise_rate: float, rows: list[dict[str, Any]]) -> dict[str, Any]:
    best_mean = max(rows, key=lambda row: row["metrics"]["mean_accuracy"])
    best_worst = max(rows, key=lambda row: row["metrics"]["worst_accuracy"])
    final_row = rows[-1]
    return {
        "noise_rate": noise_rate,
        "best_mean_step": best_mean["step"],
        "best_mean_accuracy": best_mean["metrics"]["mean_accuracy"],
        "best_worst_step": best_worst["step"],
        "best_worst_accuracy": best_worst["metrics"]["worst_accuracy"],
        "last_step": final_row["step"],
        "last_mean_accuracy": final_row["metrics"]["mean_accuracy"],
        "last_worst_accuracy": final_row["metrics"]["worst_accuracy"],
    }


def write_markdown_report(
    path: Path,
    checkpoint_summaries: list[dict[str, Any]],
    final_rows: list[dict[str, Any]],
    series_by_noise: list[tuple[float, list[dict[str, Any]]]],
) -> None:
    final_by_noise = {round(float(row["noise_rate"]), 3): row for row in final_rows}
    lines = [
        "# Noise Sweep Checkpoint Analysis",
        "",
        "Data sources:",
        "- Checkpoint curves come from training logs via `trainer.evaluate(max_batches=50)` at every saved checkpoint.",
        "- Final noise-rate comparison comes from each `noise_*/eval_metrics.json` full-validation export.",
        "",
        "Generated files:",
        "- `training_eval_checkpoints.csv`: checkpoint-by-checkpoint metrics across noise rates.",
        "- `final_eval_metrics.csv`: final full-validation metrics across noise rates.",
        "- `noise_0.1_checkpoint_training_eval.svg`, `noise_0.2_checkpoint_training_eval.svg`, `noise_0.3_checkpoint_training_eval.svg`.",
        "- `checkpoint_accuracy_by_noise.svg`, `checkpoint_loss_by_noise.svg`, `checkpoint_metric_heatmaps.svg`, `final_eval_vs_noise.svg`.",
        "",
        "Best checkpoint summary:",
        "",
        "| noise | best mean step | best mean acc | best worst step | best worst acc | last-step mean acc | final full-val mean acc |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in checkpoint_summaries:
        final_metrics = final_by_noise[round(float(row["noise_rate"]), 3)]
        lines.append(
            "| "
            f"{row['noise_rate']:.1f} | {row['best_mean_step']} | {row['best_mean_accuracy']:.3f} | "
            f"{row['best_worst_step']} | {row['best_worst_accuracy']:.3f} | "
            f"{row['last_mean_accuracy']:.3f} | {final_metrics['mean_accuracy']:.3f} |"
        )
    lines.extend(
        [
            "",
            "Notes:",
            "- The checkpoint curves and final full-validation values are related but not identical, because the checkpoint curves use the trainer's `max_batches=50` validation snapshot.",
            "- `noise_0.1` also has separate post-hoc checkpoint JSONs under `noise_0.1/evals/`, but those are not available for `0.2` and `0.3`, so they were not used for the cross-noise plots.",
        ]
    )
    ensure_dir(path.parent)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    repo_root = Path(args.repo_root).resolve()
    output_dir = repo_root / args.output_dir
    ensure_dir(output_dir)

    noise_runs = discover_noise_runs(repo_root, args.noise_rates)
    training_series: list[tuple[float, list[dict[str, Any]]]] = []
    final_metrics_series: list[tuple[float, dict[str, float]]] = []

    for run in noise_runs:
        rows = parse_training_eval_rows(run.log_path)
        final_metrics = load_final_eval_metrics(run.output_dir)
        training_series.append((run.noise_rate, rows))
        final_metrics_series.append((run.noise_rate, final_metrics))

    training_csv_rows = build_training_csv_rows(training_series)
    training_fieldnames = [
        "noise_rate",
        "step",
        "mean_accuracy",
        "worst_accuracy",
        "mean_loss",
        "worst_loss",
    ] + [f"{objective}_{suffix}" for objective in OBJECTIVES for suffix in ("accuracy", "loss")]
    write_csv(output_dir / "training_eval_checkpoints.csv", training_fieldnames, training_csv_rows)

    final_csv_rows = build_final_csv_rows(final_metrics_series)
    final_fieldnames = [
        "noise_rate",
        "mean_accuracy",
        "worst_accuracy",
        "mean_loss",
        "worst_loss",
    ] + [f"{objective}_{suffix}" for objective in OBJECTIVES for suffix in ("accuracy", "loss")]
    write_csv(output_dir / "final_eval_metrics.csv", final_fieldnames, final_csv_rows)

    for noise_rate, rows in training_series:
        write_per_noise_plot(output_dir / f"noise_{noise_tag(noise_rate)}_checkpoint_training_eval.svg", noise_rate, rows)

    write_combined_checkpoint_accuracy_plot(output_dir / "checkpoint_accuracy_by_noise.svg", training_series)
    write_combined_checkpoint_loss_plot(output_dir / "checkpoint_loss_by_noise.svg", training_series)
    write_heatmap_plot(output_dir / "checkpoint_metric_heatmaps.svg", training_series)
    write_final_eval_plot(output_dir / "final_eval_vs_noise.svg", final_csv_rows)

    checkpoint_summaries = [best_checkpoint_summary(noise_rate, rows) for noise_rate, rows in training_series]
    write_json(
        output_dir / "analysis_summary.json",
        {
            "checkpoint_summaries": checkpoint_summaries,
            "final_eval_rows": final_csv_rows,
        },
    )
    write_markdown_report(output_dir / "README.md", checkpoint_summaries, final_csv_rows, training_series)


if __name__ == "__main__":
    main()
