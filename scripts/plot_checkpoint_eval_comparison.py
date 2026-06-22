#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any


OBJECTIVES = ["helpfulness", "correctness", "coherence", "complexity", "verbosity"]
RUN_COLORS = {
    "baseline": "#1d3557",
    "candidate": "#e76f51",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare two checkpoint eval directories and create overlay plots."
    )
    parser.add_argument("--baseline_eval_dir", required=True)
    parser.add_argument("--candidate_eval_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--baseline_label", default="Baseline")
    parser.add_argument("--candidate_label", default="Candidate")
    parser.add_argument("--baseline_final_json", default=None)
    parser.add_argument("--candidate_final_json", default=None)
    return parser.parse_args()


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
    *,
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


def choose_ticks(xs: list[float], limit: int = 6) -> list[float]:
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
    xs: list[float],
    series: list[tuple[str, list[float], str]],
    *,
    x0: float,
    y0: float,
    w: float,
    h: float,
    x_label: str,
    y_label: str,
    ymin: float | None = None,
    ymax: float | None = None,
    x_tick_format: str = "{:.0f}",
    value_format: str = "{:.3f}",
) -> None:
    all_values = [value for _, values, _ in series for value in values]
    plot_ymin = min(all_values) if ymin is None else ymin
    plot_ymax = max(all_values) if ymax is None else ymax
    if math.isclose(plot_ymin, plot_ymax):
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
            f'<text x="{px:.1f}" y="{y0 + h + 18:.1f}" text-anchor="middle" font-size="11" font-family="sans-serif">{x_tick_format.format(tick)}</text>'
        )

    parts.append(
        f'<text x="{x0 + w / 2:.1f}" y="{y0 + h + 36:.1f}" text-anchor="middle" font-size="12" font-family="sans-serif">{x_label}</text>'
    )
    parts.append(
        f'<text x="{x0 - 44:.1f}" y="{y0 + h / 2:.1f}" text-anchor="middle" font-size="12" font-family="sans-serif" transform="rotate(-90 {x0 - 44:.1f},{y0 + h / 2:.1f})">{y_label}</text>'
    )

    for label, values, color in series:
        points = scale_points(
            xs,
            values,
            x0=x0,
            y0=y0,
            w=w,
            h=h,
            xmin=xmin,
            xmax=xmax,
            ymin=plot_ymin,
            ymax=plot_ymax,
        )
        parts.append(f'<polyline fill="none" stroke="{color}" stroke-width="2.5" points="{polyline(points)}"/>')
        for px, py in points:
            parts.append(f'<circle cx="{px:.2f}" cy="{py:.2f}" r="3.1" fill="{color}"/>')

    legend_x = x0 + 10
    legend_y = y0 + 18
    for index, (label, _, color) in enumerate(series):
        ly = legend_y + index * 18
        parts.append(f'<line x1="{legend_x}" y1="{ly}" x2="{legend_x + 18}" y2="{ly}" stroke="{color}" stroke-width="2.5"/>')
        parts.append(f'<text x="{legend_x + 24}" y="{ly + 4}" font-size="11" font-family="sans-serif">{label}</text>')


def bar_panel(
    parts: list[str],
    title: str,
    labels: list[str],
    baseline_values: list[float],
    candidate_values: list[float],
    *,
    baseline_label: str,
    candidate_label: str,
    x0: float,
    y0: float,
    w: float,
    h: float,
    y_label: str,
    value_format: str = "{:.3f}",
) -> None:
    finite_values = [value for value in baseline_values + candidate_values if math.isfinite(value)]
    if finite_values:
        ymin = min(0.0, min(finite_values))
        ymax = max(finite_values)
    else:
        ymin = 0.0
        ymax = 1.0
    if math.isclose(ymin, ymax):
        ymax = ymin + 1.0
    pad = (ymax - ymin) * 0.10
    ymin -= pad
    ymax += pad

    parts.append(f'<rect x="{x0}" y="{y0}" width="{w}" height="{h}" fill="white" stroke="#d7d7d7"/>')
    parts.append(
        f'<text x="{x0 + w / 2:.1f}" y="{y0 - 12:.1f}" text-anchor="middle" font-size="17" font-family="sans-serif">{title}</text>'
    )
    for frac in [0.0, 0.25, 0.5, 0.75, 1.0]:
        py = y0 + h - frac * h
        y_value = ymin + frac * (ymax - ymin)
        parts.append(f'<line x1="{x0}" y1="{py:.1f}" x2="{x0 + w}" y2="{py:.1f}" stroke="#ededed"/>')
        parts.append(
            f'<text x="{x0 - 8:.1f}" y="{py + 4:.1f}" text-anchor="end" font-size="11" font-family="sans-serif">{value_format.format(y_value)}</text>'
        )

    parts.append(
        f'<text x="{x0 - 44:.1f}" y="{y0 + h / 2:.1f}" text-anchor="middle" font-size="12" font-family="sans-serif" transform="rotate(-90 {x0 - 44:.1f},{y0 + h / 2:.1f})">{y_label}</text>'
    )

    group_width = w / len(labels)
    bar_width = group_width * 0.28
    for idx, label in enumerate(labels):
        group_x = x0 + idx * group_width
        center_x = group_x + group_width / 2
        values = [
            (baseline_values[idx], RUN_COLORS["baseline"], baseline_label, center_x - bar_width * 1.1),
            (candidate_values[idx], RUN_COLORS["candidate"], candidate_label, center_x + bar_width * 0.1),
        ]
        for value, color, _, bar_x in values:
            if not math.isfinite(value):
                parts.append(
                    f'<text x="{bar_x + bar_width / 2:.2f}" y="{y0 + 18:.2f}" text-anchor="middle" font-size="10" font-family="sans-serif" fill="{color}">nan</text>'
                )
                continue
            if ymax > ymin:
                bar_top = y0 + h - ((value - ymin) / (ymax - ymin) * h)
            else:
                bar_top = y0 + h / 2
            zero_y = y0 + h - ((0.0 - ymin) / (ymax - ymin) * h if ymax > ymin else 0.0)
            rect_y = min(bar_top, zero_y)
            rect_h = abs(zero_y - bar_top)
            parts.append(
                f'<rect x="{bar_x:.2f}" y="{rect_y:.2f}" width="{bar_width:.2f}" height="{max(rect_h, 1.0):.2f}" fill="{color}" opacity="0.9"/>'
            )
        parts.append(
            f'<text x="{center_x:.1f}" y="{y0 + h + 18:.1f}" text-anchor="middle" font-size="10" font-family="sans-serif">{label}</text>'
        )

    legend_x = x0 + 10
    legend_y = y0 + 18
    legend_items = [
        (baseline_label, RUN_COLORS["baseline"]),
        (candidate_label, RUN_COLORS["candidate"]),
    ]
    for index, (label, color) in enumerate(legend_items):
        ly = legend_y + index * 18
        parts.append(f'<rect x="{legend_x}" y="{ly - 9}" width="14" height="14" fill="{color}"/>')
        parts.append(f'<text x="{legend_x + 22}" y="{ly + 3}" font-size="11" font-family="sans-serif">{label}</text>')


def load_checkpoint_rows(eval_dir: Path) -> list[dict[str, Any]]:
    json_paths = sorted(eval_dir.glob("checkpoint-*.json"), key=lambda path: int(path.stem.split("-", 1)[1]))
    rows: list[dict[str, Any]] = []
    for path in json_paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows.append(
            {
                "step": int(path.stem.split("-", 1)[1]),
                "metrics": {key: float(value) for key, value in payload["metrics"].items()},
            }
        )
    return rows


def load_metrics_file(path: Path | None) -> dict[str, float] | None:
    if path is None:
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {key: float(value) for key, value in payload["metrics"].items()}


def align_rows(
    baseline_rows: list[dict[str, Any]], candidate_rows: list[dict[str, Any]]
) -> tuple[list[int], list[dict[str, Any]], list[dict[str, Any]]]:
    baseline_by_step = {row["step"]: row for row in baseline_rows}
    candidate_by_step = {row["step"]: row for row in candidate_rows}
    common_steps = sorted(set(baseline_by_step) & set(candidate_by_step))
    if not common_steps:
        raise ValueError("No common checkpoint steps found between the two eval directories.")
    return (
        common_steps,
        [baseline_by_step[step] for step in common_steps],
        [candidate_by_step[step] for step in common_steps],
    )


def comparison_csv_rows(
    steps: list[int], baseline_rows: list[dict[str, Any]], candidate_rows: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for step, baseline_row, candidate_row in zip(steps, baseline_rows, candidate_rows, strict=True):
        baseline_metrics = baseline_row["metrics"]
        candidate_metrics = candidate_row["metrics"]
        row: dict[str, Any] = {"step": step}
        for metric_name in ["mean_accuracy", "worst_accuracy", "mean_loss", "worst_loss"]:
            row[f"baseline_{metric_name}"] = baseline_metrics[metric_name]
            row[f"candidate_{metric_name}"] = candidate_metrics[metric_name]
            row[f"delta_{metric_name}"] = candidate_metrics[metric_name] - baseline_metrics[metric_name]
        for objective in OBJECTIVES:
            for suffix in ("accuracy", "loss"):
                metric_name = f"{objective}/{suffix}"
                row[f"baseline_{objective}_{suffix}"] = baseline_metrics[metric_name]
                row[f"candidate_{objective}_{suffix}"] = candidate_metrics[metric_name]
                row[f"delta_{objective}_{suffix}"] = candidate_metrics[metric_name] - baseline_metrics[metric_name]
        rows.append(row)
    return rows


def best_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    best_mean = max(rows, key=lambda row: row["metrics"]["mean_accuracy"])
    best_worst = max(rows, key=lambda row: row["metrics"]["worst_accuracy"])
    last_row = rows[-1]
    return {
        "best_mean_step": best_mean["step"],
        "best_mean_accuracy": best_mean["metrics"]["mean_accuracy"],
        "best_worst_step": best_worst["step"],
        "best_worst_accuracy": best_worst["metrics"]["worst_accuracy"],
        "last_step": last_row["step"],
        "last_mean_accuracy": last_row["metrics"]["mean_accuracy"],
        "last_worst_accuracy": last_row["metrics"]["worst_accuracy"],
    }


def write_aggregate_plot(
    path: Path,
    *,
    steps: list[int],
    baseline_rows: list[dict[str, Any]],
    candidate_rows: list[dict[str, Any]],
    baseline_label: str,
    candidate_label: str,
) -> None:
    xs = [float(step) for step in steps]
    width, height = 1240, 920
    panel_w, panel_h = 470, 310
    parts = svg_header(width, height, "Checkpoint aggregate comparison")
    panels = [
        ("Mean Accuracy", "mean_accuracy", "Accuracy", 70, 70, 0.30, 0.80),
        ("Worst Accuracy", "worst_accuracy", "Accuracy", 670, 70, 0.20, 0.80),
        ("Mean Loss", "mean_loss", "Loss", 70, 500, None, None),
        ("Worst Loss", "worst_loss", "Loss", 670, 500, None, None),
    ]
    for title, metric_name, y_label, x0, y0, ymin, ymax in panels:
        series = [
            (
                baseline_label,
                [row["metrics"][metric_name] for row in baseline_rows],
                RUN_COLORS["baseline"],
            ),
            (
                candidate_label,
                [row["metrics"][metric_name] for row in candidate_rows],
                RUN_COLORS["candidate"],
            ),
        ]
        line_panel(
            parts,
            title,
            xs,
            series,
            x0=x0,
            y0=y0,
            w=panel_w,
            h=panel_h,
            x_label="Checkpoint step",
            y_label=y_label,
            ymin=ymin,
            ymax=ymax,
        )
    close_svg(parts, path)


def write_objective_plot(
    path: Path,
    *,
    steps: list[int],
    baseline_rows: list[dict[str, Any]],
    candidate_rows: list[dict[str, Any]],
    baseline_label: str,
    candidate_label: str,
    suffix: str,
    y_label: str,
    title: str,
    ymin: float | None = None,
    ymax: float | None = None,
) -> None:
    xs = [float(step) for step in steps]
    width, height = 1340, 880
    panel_w, panel_h = 360, 240
    parts = svg_header(width, height, title)
    positions = [
        (70, 70),
        (490, 70),
        (910, 70),
        (280, 420),
        (700, 420),
    ]
    for objective, (x0, y0) in zip(OBJECTIVES, positions, strict=True):
        series = [
            (
                baseline_label,
                [row["metrics"][f"{objective}/{suffix}"] for row in baseline_rows],
                RUN_COLORS["baseline"],
            ),
            (
                candidate_label,
                [row["metrics"][f"{objective}/{suffix}"] for row in candidate_rows],
                RUN_COLORS["candidate"],
            ),
        ]
        line_panel(
            parts,
            objective.capitalize(),
            xs,
            series,
            x0=x0,
            y0=y0,
            w=panel_w,
            h=panel_h,
            x_label="Checkpoint step",
            y_label=y_label,
            ymin=ymin,
            ymax=ymax,
        )
    close_svg(parts, path)


def write_final_plot(
    path: Path,
    *,
    baseline_label: str,
    candidate_label: str,
    baseline_metrics: dict[str, float],
    candidate_metrics: dict[str, float],
) -> None:
    accuracy_labels = ["mean", "worst"] + OBJECTIVES
    accuracy_baseline = [baseline_metrics["mean_accuracy"], baseline_metrics["worst_accuracy"]] + [
        baseline_metrics[f"{objective}/accuracy"] for objective in OBJECTIVES
    ]
    accuracy_candidate = [candidate_metrics["mean_accuracy"], candidate_metrics["worst_accuracy"]] + [
        candidate_metrics[f"{objective}/accuracy"] for objective in OBJECTIVES
    ]
    loss_labels = ["mean", "worst"] + OBJECTIVES
    loss_baseline = [baseline_metrics["mean_loss"], baseline_metrics["worst_loss"]] + [
        baseline_metrics[f"{objective}/loss"] for objective in OBJECTIVES
    ]
    loss_candidate = [candidate_metrics["mean_loss"], candidate_metrics["worst_loss"]] + [
        candidate_metrics[f"{objective}/loss"] for objective in OBJECTIVES
    ]

    width, height = 1320, 520
    parts = svg_header(width, height, "Final full-validation comparison")
    bar_panel(
        parts,
        "Accuracy metrics",
        accuracy_labels,
        accuracy_baseline,
        accuracy_candidate,
        baseline_label=baseline_label,
        candidate_label=candidate_label,
        x0=70,
        y0=70,
        w=520,
        h=320,
        y_label="Accuracy",
    )
    bar_panel(
        parts,
        "Loss metrics",
        loss_labels,
        loss_baseline,
        loss_candidate,
        baseline_label=baseline_label,
        candidate_label=candidate_label,
        x0=710,
        y0=70,
        w=520,
        h=320,
        y_label="Loss",
    )
    close_svg(parts, path)


def write_report(
    path: Path,
    *,
    baseline_label: str,
    candidate_label: str,
    baseline_summary: dict[str, Any],
    candidate_summary: dict[str, Any],
    step_rows: list[dict[str, Any]],
    final_delta_mean_accuracy: float | None,
) -> None:
    best_step = max(step_rows, key=lambda row: row["delta_mean_accuracy"])
    last_step = step_rows[-1]
    lines = [
        "# Checkpoint Eval Comparison",
        "",
        f"- Baseline: `{baseline_label}`",
        f"- Candidate: `{candidate_label}`",
        f"- Common checkpoint count: {len(step_rows)}",
        "",
        "Checkpoint summary:",
        "",
        "| run | best mean step | best mean acc | best worst step | best worst acc | last step | last mean acc |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        f"| {baseline_label} | {baseline_summary['best_mean_step']} | {baseline_summary['best_mean_accuracy']:.3f} | {baseline_summary['best_worst_step']} | {baseline_summary['best_worst_accuracy']:.3f} | {baseline_summary['last_step']} | {baseline_summary['last_mean_accuracy']:.3f} |",
        f"| {candidate_label} | {candidate_summary['best_mean_step']} | {candidate_summary['best_mean_accuracy']:.3f} | {candidate_summary['best_worst_step']} | {candidate_summary['best_worst_accuracy']:.3f} | {candidate_summary['last_step']} | {candidate_summary['last_mean_accuracy']:.3f} |",
        "",
        f"Best checkpoint-step gain in mean accuracy for {candidate_label}: step {best_step['step']} ({best_step['delta_mean_accuracy']:+.3f}).",
        f"Last aligned checkpoint delta in mean accuracy: {last_step['delta_mean_accuracy']:+.3f}.",
    ]
    if final_delta_mean_accuracy is not None:
        lines.append(f"Final full-validation mean-accuracy delta: {final_delta_mean_accuracy:+.3f}.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    baseline_eval_dir = Path(args.baseline_eval_dir).resolve()
    candidate_eval_dir = Path(args.candidate_eval_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    ensure_dir(output_dir)

    baseline_rows = load_checkpoint_rows(baseline_eval_dir)
    candidate_rows = load_checkpoint_rows(candidate_eval_dir)
    steps, baseline_rows, candidate_rows = align_rows(baseline_rows, candidate_rows)

    rows = comparison_csv_rows(steps, baseline_rows, candidate_rows)
    fieldnames = list(rows[0].keys())
    write_csv(output_dir / "checkpoint_eval_comparison.csv", fieldnames, rows)

    baseline_summary = best_summary(baseline_rows)
    candidate_summary = best_summary(candidate_rows)

    write_aggregate_plot(
        output_dir / "checkpoint_aggregate_overlay.svg",
        steps=steps,
        baseline_rows=baseline_rows,
        candidate_rows=candidate_rows,
        baseline_label=args.baseline_label,
        candidate_label=args.candidate_label,
    )
    write_objective_plot(
        output_dir / "checkpoint_objective_accuracy_overlay.svg",
        steps=steps,
        baseline_rows=baseline_rows,
        candidate_rows=candidate_rows,
        baseline_label=args.baseline_label,
        candidate_label=args.candidate_label,
        suffix="accuracy",
        y_label="Accuracy",
        title="Checkpoint objective accuracy comparison",
        ymin=0.20,
        ymax=0.90,
    )
    write_objective_plot(
        output_dir / "checkpoint_objective_loss_overlay.svg",
        steps=steps,
        baseline_rows=baseline_rows,
        candidate_rows=candidate_rows,
        baseline_label=args.baseline_label,
        candidate_label=args.candidate_label,
        suffix="loss",
        y_label="Loss",
        title="Checkpoint objective loss comparison",
    )

    baseline_final = load_metrics_file(Path(args.baseline_final_json).resolve()) if args.baseline_final_json else None
    candidate_final = load_metrics_file(Path(args.candidate_final_json).resolve()) if args.candidate_final_json else None
    final_delta_mean_accuracy: float | None = None
    if baseline_final is not None and candidate_final is not None:
        final_delta_mean_accuracy = candidate_final["mean_accuracy"] - baseline_final["mean_accuracy"]
        write_final_plot(
            output_dir / "final_full_validation_comparison.svg",
            baseline_label=args.baseline_label,
            candidate_label=args.candidate_label,
            baseline_metrics=baseline_final,
            candidate_metrics=candidate_final,
        )

    summary_payload = {
        "baseline_label": args.baseline_label,
        "candidate_label": args.candidate_label,
        "common_steps": steps,
        "baseline_summary": baseline_summary,
        "candidate_summary": candidate_summary,
        "best_step_delta_mean_accuracy": max(rows, key=lambda row: row["delta_mean_accuracy"]),
        "last_step_delta_mean_accuracy": rows[-1]["delta_mean_accuracy"],
        "final_delta_mean_accuracy": final_delta_mean_accuracy,
        "baseline_final_metrics": baseline_final,
        "candidate_final_metrics": candidate_final,
    }
    write_json(output_dir / "summary.json", summary_payload)
    write_report(
        output_dir / "README.md",
        baseline_label=args.baseline_label,
        candidate_label=args.candidate_label,
        baseline_summary=baseline_summary,
        candidate_summary=candidate_summary,
        step_rows=rows,
        final_delta_mean_accuracy=final_delta_mean_accuracy,
    )


if __name__ == "__main__":
    main()
