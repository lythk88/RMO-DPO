#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path


OBJECTIVES = ["helpfulness", "correctness", "coherence", "complexity", "verbosity"]
COLORS = {
    "mean": "#1f77b4",
    "worst": "#d62728",
    "helpfulness": "#2ca02c",
    "correctness": "#9467bd",
    "coherence": "#8c564b",
    "complexity": "#e377c2",
    "verbosity": "#ff7f0e",
}


def scale_points(xs: list[int], ys: list[float], x0: float, y0: float, w: float, h: float, ymin: float, ymax: float) -> list[tuple[float, float]]:
    xmin, xmax = min(xs), max(xs)
    points: list[tuple[float, float]] = []
    for x, y in zip(xs, ys):
        if xmax > xmin:
            px = x0 + (x - xmin) / (xmax - xmin) * w
        else:
            px = x0 + w / 2
        if ymax > ymin:
            py = y0 + h - (y - ymin) / (ymax - ymin) * h
        else:
            py = y0 + h / 2
        points.append((px, py))
    return points


def polyline(points: list[tuple[float, float]]) -> str:
    return " ".join(f"{x:.1f},{y:.1f}" for x, y in points)


def write_summary(rows: list[tuple[int, dict[str, float]]], out_path: Path) -> None:
    with out_path.open("w", encoding="utf-8") as f:
        header = ["step", "mean_accuracy", "worst_accuracy", "mean_loss", "worst_loss"]
        for obj in OBJECTIVES:
            header += [f"{obj}_accuracy", f"{obj}_loss"]
        f.write("\t".join(header) + "\n")
        for step, metrics in rows:
            vals = [
                str(step),
                str(metrics["mean_accuracy"]),
                str(metrics["worst_accuracy"]),
                str(metrics["mean_loss"]),
                str(metrics["worst_loss"]),
            ]
            for obj in OBJECTIVES:
                vals += [str(metrics[f"{obj}/accuracy"]), str(metrics[f"{obj}/loss"])]
            f.write("\t".join(vals) + "\n")


def draw_panel(
    svg: list[str],
    series: dict[str, list[float]],
    checkpoints: list[int],
    x0: float,
    y0: float,
    panel_w: float,
    panel_h: float,
    title: str,
    y_label: str,
    keys: list[str],
    ymin: float | None = None,
    ymax: float | None = None,
) -> None:
    all_vals: list[float] = []
    for key in keys:
        all_vals.extend(series[key])
    if ymin is None:
        ymin = min(all_vals)
    if ymax is None:
        ymax = max(all_vals)
    pad = (ymax - ymin) * 0.12 if ymax > ymin else 0.05
    ymin -= pad
    ymax += pad

    svg.append(f'<rect x="{x0}" y="{y0}" width="{panel_w}" height="{panel_h}" fill="white" stroke="#cccccc"/>')
    svg.append(
        f'<text x="{x0 + panel_w / 2}" y="{y0 - 15}" text-anchor="middle" font-size="18" font-family="sans-serif">{title}</text>'
    )

    for idx in range(5):
        frac = idx / 4
        y = y0 + panel_h - frac * panel_h
        val = ymin + frac * (ymax - ymin)
        svg.append(f'<line x1="{x0}" y1="{y:.1f}" x2="{x0 + panel_w}" y2="{y:.1f}" stroke="#eeeeee"/>')
        svg.append(
            f'<text x="{x0 - 8}" y="{y + 4:.1f}" text-anchor="end" font-size="12" font-family="sans-serif">{val:.3f}</text>'
        )

    for step in checkpoints:
        frac = (step - checkpoints[0]) / (checkpoints[-1] - checkpoints[0]) if checkpoints[-1] > checkpoints[0] else 0.5
        x = x0 + frac * panel_w
        svg.append(f'<line x1="{x:.1f}" y1="{y0}" x2="{x:.1f}" y2="{y0 + panel_h}" stroke="#f5f5f5"/>')
        svg.append(
            f'<text x="{x:.1f}" y="{y0 + panel_h + 20}" text-anchor="middle" font-size="12" font-family="sans-serif">{step}</text>'
        )

    svg.append(
        f'<text x="{x0 + panel_w / 2}" y="{y0 + panel_h + 40}" text-anchor="middle" font-size="14" font-family="sans-serif">Checkpoint step</text>'
    )
    svg.append(
        f'<text x="{x0 - 50}" y="{y0 + panel_h / 2}" text-anchor="middle" font-size="14" font-family="sans-serif" transform="rotate(-90 {x0 - 50},{y0 + panel_h / 2})">{y_label}</text>'
    )

    for key in keys:
        vals = series[key]
        pts = scale_points(checkpoints, vals, x0, y0, panel_w, panel_h, ymin, ymax)
        if key.startswith("mean"):
            color = COLORS["mean"]
        elif key.startswith("worst"):
            color = COLORS["worst"]
        else:
            color = COLORS[key.split("_")[0]]
        svg.append(f'<polyline fill="none" stroke="{color}" stroke-width="2" points="{polyline(pts)}"/>')
        for px, py in pts:
            svg.append(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="3" fill="{color}"/>')

    legend_x = x0 + panel_w + 15
    legend_y = y0 + 10
    for idx, key in enumerate(keys):
        if key.startswith("mean"):
            color = COLORS["mean"]
        elif key.startswith("worst"):
            color = COLORS["worst"]
        else:
            color = COLORS[key.split("_")[0]]
        ly = legend_y + idx * 18
        svg.append(f'<line x1="{legend_x}" y1="{ly}" x2="{legend_x + 18}" y2="{ly}" stroke="{color}" stroke-width="2"/>')
        svg.append(f'<text x="{legend_x + 24}" y="{ly + 4}" font-size="12" font-family="sans-serif">{key}</text>')


def main() -> None:
    parser = argparse.ArgumentParser(description="Create TSV and SVG plots from checkpoint eval JSON files.")
    parser.add_argument("--eval_dir", required=True)
    parser.add_argument(
        "--title",
        default="Noise 0.1 checkpoint evaluation (max_batches=50)",
        help="SVG title.",
    )
    args = parser.parse_args()

    base = Path(args.eval_dir)
    json_paths = sorted(base.glob("checkpoint-*.json"), key=lambda path: int(path.stem.split("-", 1)[1]))
    if not json_paths:
        raise FileNotFoundError(f"No checkpoint JSON files found in {base}")
    checkpoints = [int(path.stem.split("-", 1)[1]) for path in json_paths]
    rows: list[tuple[int, dict[str, float]]] = []
    for step, path in zip(checkpoints, json_paths):
        with path.open() as f:
            payload = json.load(f)
        rows.append((step, payload["metrics"]))

    write_summary(rows, base / "checkpoint_eval_summary.tsv")

    series: dict[str, list[float]] = {
        "mean_accuracy": [m["mean_accuracy"] for _, m in rows],
        "worst_accuracy": [m["worst_accuracy"] for _, m in rows],
        "mean_loss": [m["mean_loss"] for _, m in rows],
        "worst_loss": [m["worst_loss"] for _, m in rows],
    }
    for obj in OBJECTIVES:
        series[f"{obj}_accuracy"] = [m[f"{obj}/accuracy"] for _, m in rows]
        series[f"{obj}_loss"] = [m[f"{obj}/loss"] for _, m in rows]

    width, height = 1200, 900
    panel_w, panel_h = 420, 300
    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="600" y="28" text-anchor="middle" font-size="22" font-family="sans-serif">{args.title}</text>',
    ]
    draw_panel(svg, series, checkpoints, 70, 70, panel_w, panel_h, "Aggregate Accuracy", "Accuracy", ["mean_accuracy", "worst_accuracy"], ymin=0.35, ymax=0.60)
    draw_panel(svg, series, checkpoints, 70, 470, panel_w, panel_h, "Aggregate Loss", "Loss", ["mean_loss", "worst_loss"])
    draw_panel(svg, series, checkpoints, 650, 70, panel_w, panel_h, "Objective Accuracy", "Accuracy", [f"{obj}_accuracy" for obj in OBJECTIVES], ymin=0.35, ymax=0.75)
    draw_panel(svg, series, checkpoints, 650, 470, panel_w, panel_h, "Objective Loss", "Loss", [f"{obj}_loss" for obj in OBJECTIVES])
    svg.append("</svg>")

    (base / "checkpoint_eval_plot.svg").write_text("\n".join(svg), encoding="utf-8")


if __name__ == "__main__":
    main()
