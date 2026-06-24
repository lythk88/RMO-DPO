#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import html
import json
import math
from pathlib import Path
from typing import Any


OBJECTIVES = ["helpfulness", "correctness", "coherence", "complexity", "verbosity"]
BASELINE_COLOR = "#1d3557"
CANDIDATE_COLOR = "#e76f51"
DELTA_POSITIVE = "#2a9d8f"
DELTA_NEGATIVE = "#c44536"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot a final-eval comparison for two HelpSteer2 eval JSONs.")
    parser.add_argument("--baseline_json", required=True)
    parser.add_argument("--candidate_json", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--baseline_label", default="Baseline")
    parser.add_argument("--candidate_label", default="Candidate")
    parser.add_argument("--title", default="Final HelpSteer2 Validation Comparison")
    return parser.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_metrics(path: Path) -> dict[str, float]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {key: float(value) for key, value in payload["metrics"].items()}


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: Any) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def metric_rows(baseline: dict[str, float], candidate: dict[str, float]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    metrics = [
        ("mean_accuracy", "aggregate", "accuracy", "mean"),
        ("worst_accuracy", "aggregate", "accuracy", "worst"),
        ("mean_loss", "aggregate", "loss", "mean"),
        ("worst_loss", "aggregate", "loss", "worst"),
    ]
    for objective in OBJECTIVES:
        metrics.extend(
            [
                (f"{objective}/accuracy", objective, "accuracy", objective),
                (f"{objective}/loss", objective, "loss", objective),
                (f"{objective}/margin", objective, "margin", objective),
            ]
        )

    for key, group, metric_type, label in metrics:
        base_value = baseline[key]
        candidate_value = candidate[key]
        rows.append(
            {
                "metric": key,
                "group": group,
                "metric_type": metric_type,
                "label": label,
                "baseline": base_value,
                "candidate": candidate_value,
                "delta": candidate_value - base_value,
            }
        )
    return rows


def fmt(value: float, digits: int = 3) -> str:
    return f"{value:.{digits}f}"


def fmt_signed(value: float, digits: int = 3) -> str:
    return f"{value:+.{digits}f}"


def esc(text: str) -> str:
    return html.escape(text, quote=True)


def safe_range(values: list[float], pad_fraction: float = 0.10) -> tuple[float, float]:
    finite = [value for value in values if math.isfinite(value)]
    if not finite:
        return 0.0, 1.0
    lo, hi = min(finite), max(finite)
    if math.isclose(lo, hi):
        delta = max(abs(lo) * 0.05, 0.01)
        return lo - delta, hi + delta
    pad = (hi - lo) * pad_fraction
    return lo - pad, hi + pad


def scale_x(value: float, x0: float, width: float, xmin: float, xmax: float) -> float:
    if math.isclose(xmin, xmax):
        return x0 + width / 2
    return x0 + (value - xmin) / (xmax - xmin) * width


def scale_y(value: float, y0: float, height: float, ymin: float, ymax: float) -> float:
    if math.isclose(ymin, ymax):
        return y0 + height / 2
    return y0 + height - (value - ymin) / (ymax - ymin) * height


def svg_header(width: int, height: int, title: str) -> list[str]:
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#fbfaf7"/>',
        f'<text x="{width / 2:.1f}" y="34" text-anchor="middle" font-size="24" font-family="sans-serif" font-weight="700">{esc(title)}</text>',
    ]


def panel(parts: list[str], title: str, x0: float, y0: float, width: float, height: float) -> None:
    parts.append(f'<rect x="{x0}" y="{y0}" width="{width}" height="{height}" fill="white" stroke="#d8d8d8"/>')
    parts.append(
        f'<text x="{x0 + width / 2:.1f}" y="{y0 - 14:.1f}" text-anchor="middle" font-size="17" font-family="sans-serif" font-weight="700">{esc(title)}</text>'
    )


def y_grid(
    parts: list[str],
    x0: float,
    y0: float,
    width: float,
    height: float,
    ymin: float,
    ymax: float,
    *,
    digits: int = 3,
) -> None:
    for idx in range(5):
        frac = idx / 4
        y = y0 + height - frac * height
        value = ymin + frac * (ymax - ymin)
        parts.append(f'<line x1="{x0}" y1="{y:.1f}" x2="{x0 + width}" y2="{y:.1f}" stroke="#eeeeee"/>')
        parts.append(
            f'<text x="{x0 - 8:.1f}" y="{y + 4:.1f}" text-anchor="end" font-size="11" font-family="sans-serif">{fmt(value, digits)}</text>'
        )


def x_grid(
    parts: list[str],
    x0: float,
    y0: float,
    width: float,
    height: float,
    xmin: float,
    xmax: float,
    *,
    digits: int = 3,
) -> None:
    for idx in range(5):
        frac = idx / 4
        x = x0 + frac * width
        value = xmin + frac * (xmax - xmin)
        parts.append(f'<line x1="{x:.1f}" y1="{y0}" x2="{x:.1f}" y2="{y0 + height}" stroke="#eeeeee"/>')
        parts.append(
            f'<text x="{x:.1f}" y="{y0 + height + 18:.1f}" text-anchor="middle" font-size="11" font-family="sans-serif">{fmt(value, digits)}</text>'
        )


def legend(parts: list[str], x: float, y: float, baseline_label: str, candidate_label: str) -> None:
    items = [(baseline_label, BASELINE_COLOR), (candidate_label, CANDIDATE_COLOR)]
    for idx, (label, color) in enumerate(items):
        ly = y + idx * 19
        parts.append(f'<rect x="{x}" y="{ly - 10}" width="14" height="14" fill="{color}"/>')
        parts.append(f'<text x="{x + 22}" y="{ly + 2}" font-size="12" font-family="sans-serif">{esc(label)}</text>')


def paired_bar_panel(
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
    width: float,
    height: float,
    ymin: float | None = None,
    ymax: float | None = None,
    digits: int = 3,
) -> None:
    values = baseline_values + candidate_values
    data_min, data_max = safe_range(values)
    ymin = data_min if ymin is None else ymin
    ymax = data_max if ymax is None else ymax
    panel(parts, title, x0, y0, width, height)
    y_grid(parts, x0, y0, width, height, ymin, ymax, digits=digits)
    legend(parts, x0 + 10, y0 + 20, baseline_label, candidate_label)

    group_width = width / len(labels)
    bar_width = group_width * 0.26
    if ymin <= 0.0 <= ymax:
        bar_base_value = 0.0
    elif min(values) >= 0.0:
        bar_base_value = ymin
    else:
        bar_base_value = ymax
    bar_base_y = scale_y(bar_base_value, y0, height, ymin, ymax)
    for idx, label in enumerate(labels):
        center = x0 + idx * group_width + group_width / 2
        for value, color, bar_x in [
            (baseline_values[idx], BASELINE_COLOR, center - bar_width * 1.1),
            (candidate_values[idx], CANDIDATE_COLOR, center + bar_width * 0.1),
        ]:
            top_y = scale_y(value, y0, height, ymin, ymax)
            rect_y = min(top_y, bar_base_y)
            rect_h = max(abs(bar_base_y - top_y), 1.0)
            parts.append(
                f'<rect x="{bar_x:.1f}" y="{rect_y:.1f}" width="{bar_width:.1f}" height="{rect_h:.1f}" fill="{color}" opacity="0.92"/>'
            )
            parts.append(
                f'<text x="{bar_x + bar_width / 2:.1f}" y="{top_y - 5:.1f}" text-anchor="middle" font-size="10" font-family="sans-serif">{fmt(value, digits)}</text>'
            )
        delta = candidate_values[idx] - baseline_values[idx]
        parts.append(
            f'<text x="{center:.1f}" y="{y0 + height + 36:.1f}" text-anchor="middle" font-size="11" font-family="sans-serif" fill="{DELTA_POSITIVE if delta >= 0 else DELTA_NEGATIVE}">{fmt_signed(delta, digits)}</text>'
        )
        parts.append(
            f'<text x="{center:.1f}" y="{y0 + height + 18:.1f}" text-anchor="middle" font-size="11" font-family="sans-serif">{esc(label)}</text>'
        )


def dumbbell_panel(
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
    width: float,
    height: float,
    xmin: float | None = None,
    xmax: float | None = None,
    digits: int = 3,
    lower_is_better: bool = False,
) -> None:
    data_min, data_max = safe_range(baseline_values + candidate_values, pad_fraction=0.15)
    xmin = data_min if xmin is None else xmin
    xmax = data_max if xmax is None else xmax
    panel(parts, title, x0, y0, width, height)
    x_grid(parts, x0, y0, width, height, xmin, xmax, digits=digits)
    legend(parts, x0 + width - 170, y0 + 20, baseline_label, candidate_label)

    row_gap = height / len(labels)
    for idx, label in enumerate(labels):
        center_y = y0 + idx * row_gap + row_gap / 2
        bx = scale_x(baseline_values[idx], x0, width, xmin, xmax)
        cx = scale_x(candidate_values[idx], x0, width, xmin, xmax)
        delta = candidate_values[idx] - baseline_values[idx]
        good = delta <= 0 if lower_is_better else delta >= 0
        delta_color = DELTA_POSITIVE if good else DELTA_NEGATIVE
        parts.append(f'<line x1="{x0}" y1="{center_y:.1f}" x2="{x0 + width}" y2="{center_y:.1f}" stroke="#f4f4f4"/>')
        parts.append(f'<line x1="{bx:.1f}" y1="{center_y:.1f}" x2="{cx:.1f}" y2="{center_y:.1f}" stroke="{delta_color}" stroke-width="2.2"/>')
        parts.append(f'<circle cx="{bx:.1f}" cy="{center_y:.1f}" r="5" fill="{BASELINE_COLOR}"/>')
        parts.append(f'<circle cx="{cx:.1f}" cy="{center_y:.1f}" r="5" fill="{CANDIDATE_COLOR}"/>')
        parts.append(
            f'<text x="{x0 - 10:.1f}" y="{center_y + 4:.1f}" text-anchor="end" font-size="12" font-family="sans-serif">{esc(label)}</text>'
        )
        parts.append(
            f'<text x="{x0 + width + 10:.1f}" y="{center_y + 4:.1f}" font-size="12" font-family="sans-serif" fill="{delta_color}">{fmt_signed(delta, digits)}</text>'
        )
    parts.append(
        f'<text x="{x0 + width / 2:.1f}" y="{y0 + height + 38:.1f}" text-anchor="middle" font-size="12" font-family="sans-serif">Metric value</text>'
    )


def delta_bar_panel(
    parts: list[str],
    title: str,
    labels: list[str],
    deltas: list[float],
    *,
    x0: float,
    y0: float,
    width: float,
    height: float,
    digits: int = 3,
    lower_is_better: bool = False,
) -> None:
    abs_max = max(max(abs(value) for value in deltas), 0.01)
    xmin, xmax = -abs_max * 1.15, abs_max * 1.15
    panel(parts, title, x0, y0, width, height)
    x_grid(parts, x0, y0, width, height, xmin, xmax, digits=digits)
    zero_x = scale_x(0.0, x0, width, xmin, xmax)
    parts.append(f'<line x1="{zero_x:.1f}" y1="{y0}" x2="{zero_x:.1f}" y2="{y0 + height}" stroke="#333333" stroke-width="1.2"/>')

    row_gap = height / len(labels)
    for idx, label in enumerate(labels):
        center_y = y0 + idx * row_gap + row_gap / 2
        delta = deltas[idx]
        good = delta <= 0 if lower_is_better else delta >= 0
        color = DELTA_POSITIVE if good else DELTA_NEGATIVE
        end_x = scale_x(delta, x0, width, xmin, xmax)
        rect_x = min(zero_x, end_x)
        rect_w = max(abs(end_x - zero_x), 1.0)
        parts.append(f'<rect x="{rect_x:.1f}" y="{center_y - 8:.1f}" width="{rect_w:.1f}" height="16" fill="{color}" opacity="0.9"/>')
        parts.append(
            f'<text x="{x0 - 10:.1f}" y="{center_y + 4:.1f}" text-anchor="end" font-size="12" font-family="sans-serif">{esc(label)}</text>'
        )
        label_x = end_x + 6 if delta >= 0 else end_x - 6
        anchor = "start" if delta >= 0 else "end"
        parts.append(
            f'<text x="{label_x:.1f}" y="{center_y + 4:.1f}" text-anchor="{anchor}" font-size="12" font-family="sans-serif">{fmt_signed(delta, digits)}</text>'
        )
    parts.append(
        f'<text x="{x0 + width / 2:.1f}" y="{y0 + height + 38:.1f}" text-anchor="middle" font-size="12" font-family="sans-serif">Candidate minus baseline</text>'
    )


def write_svg(
    path: Path,
    *,
    title: str,
    baseline_label: str,
    candidate_label: str,
    baseline: dict[str, float],
    candidate: dict[str, float],
) -> None:
    width, height = 1500, 1060
    parts = svg_header(width, height, title)

    aggregate_labels = ["mean acc", "worst acc"]
    paired_bar_panel(
        parts,
        "Aggregate Accuracy",
        aggregate_labels,
        [baseline["mean_accuracy"], baseline["worst_accuracy"]],
        [candidate["mean_accuracy"], candidate["worst_accuracy"]],
        baseline_label=baseline_label,
        candidate_label=candidate_label,
        x0=80,
        y0=90,
        width=390,
        height=270,
        ymin=0.35,
        ymax=0.60,
    )
    paired_bar_panel(
        parts,
        "Aggregate Loss",
        ["mean loss", "worst loss"],
        [baseline["mean_loss"], baseline["worst_loss"]],
        [candidate["mean_loss"], candidate["worst_loss"]],
        baseline_label=baseline_label,
        candidate_label=candidate_label,
        x0=80,
        y0=480,
        width=390,
        height=270,
        ymin=0.675,
        ymax=0.705,
    )

    objective_labels = [objective[:4] if objective != "complexity" else "compl" for objective in OBJECTIVES]
    baseline_acc = [baseline[f"{objective}/accuracy"] for objective in OBJECTIVES]
    candidate_acc = [candidate[f"{objective}/accuracy"] for objective in OBJECTIVES]
    baseline_loss = [baseline[f"{objective}/loss"] for objective in OBJECTIVES]
    candidate_loss = [candidate[f"{objective}/loss"] for objective in OBJECTIVES]

    dumbbell_panel(
        parts,
        "Objective Accuracy",
        objective_labels,
        baseline_acc,
        candidate_acc,
        baseline_label=baseline_label,
        candidate_label=candidate_label,
        x0=660,
        y0=90,
        width=500,
        height=270,
        xmin=0.38,
        xmax=0.63,
    )
    dumbbell_panel(
        parts,
        "Objective Loss",
        objective_labels,
        baseline_loss,
        candidate_loss,
        baseline_label=baseline_label,
        candidate_label=candidate_label,
        x0=660,
        y0=480,
        width=500,
        height=270,
        xmin=0.675,
        xmax=0.705,
        lower_is_better=True,
    )

    delta_bar_panel(
        parts,
        "Accuracy Delta",
        ["mean", "worst"] + objective_labels,
        [candidate["mean_accuracy"] - baseline["mean_accuracy"], candidate["worst_accuracy"] - baseline["worst_accuracy"]]
        + [candidate[f"{objective}/accuracy"] - baseline[f"{objective}/accuracy"] for objective in OBJECTIVES],
        x0=80,
        y0=870,
        width=520,
        height=130,
    )
    delta_bar_panel(
        parts,
        "Loss Delta",
        ["mean", "worst"] + objective_labels,
        [candidate["mean_loss"] - baseline["mean_loss"], candidate["worst_loss"] - baseline["worst_loss"]]
        + [candidate[f"{objective}/loss"] - baseline[f"{objective}/loss"] for objective in OBJECTIVES],
        x0=780,
        y0=870,
        width=520,
        height=130,
        lower_is_better=True,
    )

    note = "Green deltas indicate better candidate values: higher accuracy or lower loss."
    parts.append(f'<text x="{width / 2:.1f}" y="1038" text-anchor="middle" font-size="12" font-family="sans-serif" fill="#555555">{esc(note)}</text>')
    parts.append("</svg>")
    ensure_dir(path.parent)
    path.write_text("\n".join(parts), encoding="utf-8")


def write_report(
    path: Path,
    *,
    baseline_json: Path,
    candidate_json: Path,
    baseline_label: str,
    candidate_label: str,
    baseline: dict[str, float],
    candidate: dict[str, float],
) -> None:
    lines = [
        "# Final Eval Comparison",
        "",
        f"- Baseline: `{baseline_label}` from `{baseline_json}`",
        f"- Candidate: `{candidate_label}` from `{candidate_json}`",
        "",
        "| metric | baseline | candidate | delta |",
        "| --- | ---: | ---: | ---: |",
        f"| mean accuracy | {baseline['mean_accuracy']:.4f} | {candidate['mean_accuracy']:.4f} | {candidate['mean_accuracy'] - baseline['mean_accuracy']:+.4f} |",
        f"| worst accuracy | {baseline['worst_accuracy']:.4f} | {candidate['worst_accuracy']:.4f} | {candidate['worst_accuracy'] - baseline['worst_accuracy']:+.4f} |",
        f"| mean loss | {baseline['mean_loss']:.4f} | {candidate['mean_loss']:.4f} | {candidate['mean_loss'] - baseline['mean_loss']:+.4f} |",
        f"| worst loss | {baseline['worst_loss']:.4f} | {candidate['worst_loss']:.4f} | {candidate['worst_loss'] - baseline['worst_loss']:+.4f} |",
        "",
        "Objective accuracy deltas:",
        "",
        "| objective | baseline | candidate | delta |",
        "| --- | ---: | ---: | ---: |",
    ]
    for objective in OBJECTIVES:
        base_value = baseline[f"{objective}/accuracy"]
        candidate_value = candidate[f"{objective}/accuracy"]
        lines.append(f"| {objective} | {base_value:.4f} | {candidate_value:.4f} | {candidate_value - base_value:+.4f} |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    baseline_json = Path(args.baseline_json).resolve()
    candidate_json = Path(args.candidate_json).resolve()
    output_dir = Path(args.output_dir).resolve()
    ensure_dir(output_dir)

    baseline = load_metrics(baseline_json)
    candidate = load_metrics(candidate_json)
    rows = metric_rows(baseline, candidate)

    write_csv(output_dir / "final_eval_comparison.csv", rows)
    write_json(
        output_dir / "summary.json",
        {
            "baseline_json": str(baseline_json),
            "candidate_json": str(candidate_json),
            "baseline_label": args.baseline_label,
            "candidate_label": args.candidate_label,
            "baseline_metrics": baseline,
            "candidate_metrics": candidate,
            "deltas": {row["metric"]: row["delta"] for row in rows},
        },
    )
    write_svg(
        output_dir / "final_eval_comparison.svg",
        title=args.title,
        baseline_label=args.baseline_label,
        candidate_label=args.candidate_label,
        baseline=baseline,
        candidate=candidate,
    )
    write_report(
        output_dir / "README.md",
        baseline_json=baseline_json,
        candidate_json=candidate_json,
        baseline_label=args.baseline_label,
        candidate_label=args.candidate_label,
        baseline=baseline,
        candidate=candidate,
    )


if __name__ == "__main__":
    main()
