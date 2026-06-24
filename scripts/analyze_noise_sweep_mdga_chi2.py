#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

from wandb.proto.wandb_internal_pb2 import Record
from wandb.sdk.internal.datastore import DataStore


OBJECTIVES = ["helpfulness", "correctness", "coherence", "complexity", "verbosity"]
RUN_COLOR_BY_NAME = {
    "noise_0.0": "#1d3557",
    "noise_0.0_lambda_0.1": "#4e79a7",
    "noise_0.1": "#e76f51",
    "noise_0.1_lambda_0.1": "#f4a261",
    "noise_0.2": "#2a9d8f",
    "noise_0.2_lambda_0.1": "#52b788",
    "noise_0.3": "#6a4c93",
    "noise_0.3_lambda_0.1": "#b07bac",
}
RUN_RE = re.compile(r"noise_(\d+(?:\.\d+)?)(?:_lambda_(\d+(?:\.\d+)?))?$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze mdga-chi2 noise-sweep RMO-DPO checkpoints using local WandB histories and eval JSONs."
    )
    parser.add_argument(
        "--repo_root",
        default=str(Path(__file__).resolve().parents[1]),
        help="Repository root.",
    )
    parser.add_argument(
        "--sweep_dir",
        default="outputs/noise_sweep_mdga_chi2",
        help="Sweep directory relative to repo_root.",
    )
    parser.add_argument(
        "--output_dir",
        default="outputs/noise_sweep_mdga_chi2/analysis",
        help="Analysis output directory relative to repo_root.",
    )
    return parser.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: Any) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def parse_json_value(raw: str) -> Any:
    try:
        return json.loads(raw)
    except Exception:
        return raw


def parse_run_name(run_name: str) -> tuple[float, float | None]:
    match = RUN_RE.match(run_name)
    if not match:
        raise ValueError(f"Could not parse run name: {run_name}")
    noise_rate = float(match.group(1))
    lambda_rate = float(match.group(2)) if match.group(2) is not None else None
    return noise_rate, lambda_rate


def parse_eval_json(eval_path: Path) -> dict[str, float]:
    payload = json.loads(eval_path.read_text(encoding="utf-8"))
    metrics = payload.get("metrics", {})
    return {
        "eval_mean_accuracy": float(metrics["mean_accuracy"]),
        "eval_mean_loss": float(metrics["mean_loss"]),
        "eval_worst_accuracy": float(metrics["worst_accuracy"]),
        "eval_worst_loss": float(metrics["worst_loss"]),
    }


def parse_wandb_history_items(items: Any) -> dict[str, Any]:
    row: dict[str, Any] = {}
    for item in items:
        key = "/".join(item.nested_key) if item.nested_key else item.key
        row[key] = parse_json_value(item.value_json)
    return row


def scan_wandb_run_file(run_file: Path) -> dict[str, Any] | None:
    ds = DataStore()
    ds.open_for_scan(str(run_file))

    output_dir_rel: str | None = None
    run_name: str | None = None
    history_by_step: dict[int, dict[str, Any]] = {}

    while True:
        data = ds.scan_data()
        if data is None:
            break
        record = Record()
        record.ParseFromString(data)
        kind = record.WhichOneof("record_type")

        if kind == "run" and output_dir_rel is None:
            config = {
                update.key: parse_json_value(update.value_json)
                for update in record.run.config.update
            }
            output_dir_rel = config.get("output_dir")
            run_name = config.get("run_name")
            if not isinstance(output_dir_rel, str):
                output_dir_rel = None
            if not isinstance(run_name, str):
                run_name = None
        elif kind == "history":
            history_row = parse_wandb_history_items(record.history.item)
            step_value = history_row.get("step", history_row.get("_step"))
            if step_value is None:
                continue
            try:
                step = int(round(float(step_value)))
            except Exception:
                continue
            history_by_step.setdefault(step, {}).update(history_row)

    if output_dir_rel is None or "outputs/noise_sweep_mdga_chi2/" not in output_dir_rel:
        return None
    return {
        "run_file": run_file,
        "output_dir_rel": output_dir_rel,
        "run_name": run_name,
        "history_by_step": history_by_step,
    }


def discover_wandb_histories(repo_root: Path) -> dict[str, list[dict[str, Any]]]:
    by_output_dir: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for run_file in sorted((repo_root / "wandb").rglob("*.wandb")):
        parsed = scan_wandb_run_file(run_file)
        if parsed is None:
            continue
        by_output_dir[parsed["output_dir_rel"]].append(parsed)
    for output_dir_rel in by_output_dir:
        by_output_dir[output_dir_rel].sort(key=lambda item: item["run_file"].parent.name)
    return by_output_dir


def merge_history_runs(run_files: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    combined: dict[int, dict[str, Any]] = {}
    for run in run_files:
        for step, row in run["history_by_step"].items():
            combined.setdefault(step, {}).update(row)
    return combined


def load_eval_rows(run_dir: Path, checkpoint_steps: list[int]) -> dict[int, dict[str, Any]]:
    eval_rows: dict[int, dict[str, Any]] = {}
    eval_dir = run_dir / "evals_helpsteer2"
    if not eval_dir.exists():
        return eval_rows

    for eval_path in sorted(eval_dir.glob("checkpoint-*.json")):
        try:
            step = int(eval_path.stem.split("-")[-1])
        except Exception:
            continue
        metrics = parse_eval_json(eval_path)
        eval_rows[step] = {
            **metrics,
            "eval_source": f"json:{eval_path.name}",
        }

    max_checkpoint = max(checkpoint_steps) if checkpoint_steps else None
    for name in ["final.json", "final_retry.json"]:
        final_path = eval_dir / name
        if final_path.exists() and max_checkpoint is not None:
            metrics = parse_eval_json(final_path)
            eval_rows.setdefault(
                max_checkpoint,
                {
                    **metrics,
                    "eval_source": f"json:{final_path.name}",
                },
            )
    return eval_rows


def mean_or_none(values: list[float]) -> float | None:
    finite_values = [value for value in values if value is not None]
    if not finite_values:
        return None
    return sum(finite_values) / len(finite_values)


def last_history_step_at_or_before(steps: list[int], checkpoint_step: int) -> int | None:
    candidates = [step for step in steps if step <= checkpoint_step]
    return max(candidates) if candidates else None


def build_checkpoint_rows(repo_root: Path, sweep_dir: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    wandb_histories = discover_wandb_histories(repo_root)
    checkpoint_rows: list[dict[str, Any]] = []
    manifest_rows: list[dict[str, Any]] = []

    for run_dir in sorted(path for path in sweep_dir.iterdir() if path.is_dir()):
        run_name = run_dir.name
        noise_rate, lambda_rate = parse_run_name(run_name)
        checkpoint_steps = sorted(
            int(path.name.split("-")[-1])
            for path in run_dir.glob("checkpoint-*")
            if path.is_dir() and path.name.split("-")[-1].isdigit()
        )
        output_dir_rel = str(run_dir.relative_to(repo_root))
        history_runs = wandb_histories.get(output_dir_rel, [])
        merged_history = merge_history_runs(history_runs)
        history_steps = sorted(merged_history)
        eval_rows = load_eval_rows(run_dir, checkpoint_steps)

        notes: list[str] = []
        if not history_runs:
            notes.append("no local .wandb history")
        if not eval_rows:
            notes.append("no eval JSONs")

        manifest_rows.append(
            {
                "run_name": run_name,
                "noise_rate": noise_rate,
                "lambda_rate": lambda_rate if lambda_rate is not None else "",
                "checkpoint_count": len(checkpoint_steps),
                "wandb_run_count": len(history_runs),
                "history_step_count": len(history_steps),
                "eval_json_count": len(eval_rows),
                "notes": "; ".join(notes),
            }
        )

        for checkpoint_step in checkpoint_steps:
            history_step = last_history_step_at_or_before(history_steps, checkpoint_step)
            history_row = merged_history.get(history_step) if history_step is not None else None

            train_losses = []
            train_accuracies = []
            if history_row is not None:
                for objective in OBJECTIVES:
                    loss_value = history_row.get(f"train/{objective}_loss")
                    if isinstance(loss_value, (int, float)):
                        train_losses.append(float(loss_value))
                    acc_value = history_row.get(f"train/{objective}_acc")
                    if isinstance(acc_value, (int, float)):
                        train_accuracies.append(float(acc_value))

            checkpoint_row: dict[str, Any] = {
                "run_name": run_name,
                "noise_rate": noise_rate,
                "lambda_rate": lambda_rate if lambda_rate is not None else "",
                "checkpoint": checkpoint_step,
                "checkpoint_dir": str(run_dir / f"checkpoint-{checkpoint_step}"),
                "train_step_used": history_step if history_step is not None else "",
                "train_mean_loss": mean_or_none(train_losses),
                "train_mean_accuracy": mean_or_none(train_accuracies),
                "train_source": "wandb" if train_losses else "",
                "eval_mean_loss": None,
                "eval_mean_accuracy": None,
                "eval_worst_loss": None,
                "eval_worst_accuracy": None,
                "eval_source": "",
            }

            if history_row is not None:
                eval_mean_loss = history_row.get("eval/mean_loss")
                eval_mean_acc = history_row.get("eval/mean_acc")
                eval_worst_loss = history_row.get("eval/worst_loss")
                eval_worst_acc = history_row.get("eval/worst_acc")
                if isinstance(eval_mean_loss, (int, float)):
                    checkpoint_row["eval_mean_loss"] = float(eval_mean_loss)
                    checkpoint_row["eval_source"] = "wandb"
                if isinstance(eval_mean_acc, (int, float)):
                    checkpoint_row["eval_mean_accuracy"] = float(eval_mean_acc)
                    checkpoint_row["eval_source"] = "wandb"
                if isinstance(eval_worst_loss, (int, float)):
                    checkpoint_row["eval_worst_loss"] = float(eval_worst_loss)
                if isinstance(eval_worst_acc, (int, float)):
                    checkpoint_row["eval_worst_accuracy"] = float(eval_worst_acc)

            if checkpoint_step in eval_rows and checkpoint_row["eval_mean_accuracy"] is None:
                checkpoint_row.update(eval_rows[checkpoint_step])

            checkpoint_rows.append(checkpoint_row)

    checkpoint_rows.sort(key=lambda row: (row["noise_rate"], row["lambda_rate"] if row["lambda_rate"] != "" else -1.0, row["checkpoint"]))
    manifest_rows.sort(key=lambda row: (row["noise_rate"], row["lambda_rate"] if row["lambda_rate"] != "" else -1.0))
    return checkpoint_rows, manifest_rows


def summarize_runs(checkpoint_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_run: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in checkpoint_rows:
        by_run[row["run_name"]].append(row)

    summary_rows: list[dict[str, Any]] = []
    for run_name, rows in sorted(by_run.items(), key=lambda item: parse_run_name(item[0])):
        rows = sorted(rows, key=lambda row: row["checkpoint"])
        eval_rows = [row for row in rows if isinstance(row["eval_mean_accuracy"], (int, float))]
        train_rows = [row for row in rows if isinstance(row["train_mean_loss"], (int, float))]

        best_eval_row = max(eval_rows, key=lambda row: row["eval_mean_accuracy"]) if eval_rows else None
        best_eval_loss_row = min(eval_rows, key=lambda row: row["eval_mean_loss"]) if eval_rows else None
        final_row = rows[-1]

        notes: list[str] = []
        if not train_rows:
            notes.append("train loss unavailable")
        if not eval_rows:
            notes.append("eval metrics unavailable")
        elif len(eval_rows) < len(rows):
            notes.append("partial eval coverage")

        summary_rows.append(
            {
                "run_name": run_name,
                "noise_rate": rows[0]["noise_rate"],
                "lambda_rate": rows[0]["lambda_rate"],
                "checkpoint_count": len(rows),
                "train_metric_points": len(train_rows),
                "eval_metric_points": len(eval_rows),
                "best_eval_checkpoint": best_eval_row["checkpoint"] if best_eval_row else "",
                "best_eval_mean_accuracy": best_eval_row["eval_mean_accuracy"] if best_eval_row else "",
                "best_eval_loss_checkpoint": best_eval_loss_row["checkpoint"] if best_eval_loss_row else "",
                "best_eval_mean_loss": best_eval_loss_row["eval_mean_loss"] if best_eval_loss_row else "",
                "final_checkpoint": final_row["checkpoint"],
                "final_train_mean_loss": final_row["train_mean_loss"] if isinstance(final_row["train_mean_loss"], (int, float)) else "",
                "final_eval_mean_accuracy": final_row["eval_mean_accuracy"] if isinstance(final_row["eval_mean_accuracy"], (int, float)) else "",
                "final_eval_mean_loss": final_row["eval_mean_loss"] if isinstance(final_row["eval_mean_loss"], (int, float)) else "",
                "notes": "; ".join(notes),
            }
        )

    summary_rows.sort(key=lambda row: (row["noise_rate"], row["lambda_rate"] if row["lambda_rate"] != "" else -1.0))
    return summary_rows


def svg_polyline(points: list[tuple[float, float]]) -> str:
    return " ".join(f"{x:.2f},{y:.2f}" for x, y in points)


def svg_line_segments(points: list[tuple[float, float | None]]) -> list[list[tuple[float, float]]]:
    segments: list[list[tuple[float, float]]] = []
    current: list[tuple[float, float]] = []
    for x_value, y_value in points:
        if y_value is None or not math.isfinite(y_value):
            if current:
                segments.append(current)
                current = []
            continue
        current.append((x_value, y_value))
    if current:
        segments.append(current)
    return segments


def choose_ticks(xs: list[int], limit: int = 6) -> list[int]:
    if len(xs) <= limit:
        return xs
    idxs = {0, len(xs) - 1}
    for index in range(1, limit - 1):
        idx = round(index * (len(xs) - 1) / (limit - 1))
        idxs.add(idx)
    return [xs[idx] for idx in sorted(idxs)]


def render_panel(
    parts: list[str],
    *,
    title: str,
    series: list[tuple[str, list[tuple[int, float | None]], str]],
    x0: float,
    y0: float,
    width: float,
    height: float,
    y_label: str,
    notes: list[str],
) -> None:
    x_left = x0 + 68
    y_top = y0 + 28
    plot_width = width - 88
    plot_height = height - 56

    xs = sorted({x for _, points, _ in series for x, _ in points})
    finite_values = [
        y for _, points, _ in series for _, y in points if y is not None and math.isfinite(y)
    ]
    if not xs or not finite_values:
        parts.append(f'<rect x="{x_left}" y="{y_top}" width="{plot_width}" height="{plot_height}" fill="white" stroke="#d7d7d7"/>')
        parts.append(f'<text x="{x0 + width / 2:.1f}" y="{y0 + 16:.1f}" text-anchor="middle" font-size="16" font-family="sans-serif">{title}</text>')
        parts.append(f'<text x="{x0 + width / 2:.1f}" y="{y0 + height / 2:.1f}" text-anchor="middle" font-size="12" font-family="sans-serif">No data</text>')
        return

    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(finite_values), max(finite_values)
    if math.isclose(ymin, ymax):
        ymax = ymin + 1.0
    pad = (ymax - ymin) * 0.08
    ymin -= pad
    ymax += pad

    def sx(value: float) -> float:
        if xmax == xmin:
            return x_left + plot_width / 2
        return x_left + (value - xmin) / (xmax - xmin) * plot_width

    def sy(value: float) -> float:
        return y_top + plot_height - (value - ymin) / (ymax - ymin) * plot_height

    parts.append(f'<text x="{x0 + width / 2:.1f}" y="{y0 + 16:.1f}" text-anchor="middle" font-size="16" font-family="sans-serif">{title}</text>')
    parts.append(f'<rect x="{x_left}" y="{y_top}" width="{plot_width}" height="{plot_height}" fill="white" stroke="#d7d7d7"/>')

    for frac in [0.0, 0.25, 0.5, 0.75, 1.0]:
        py = y_top + plot_height - frac * plot_height
        y_value = ymin + frac * (ymax - ymin)
        parts.append(f'<line x1="{x_left}" y1="{py:.1f}" x2="{x_left + plot_width}" y2="{py:.1f}" stroke="#ececec"/>')
        parts.append(f'<text x="{x_left - 8:.1f}" y="{py + 4:.1f}" text-anchor="end" font-size="11" font-family="sans-serif">{y_value:.3f}</text>')

    for tick in choose_ticks(xs):
        px = sx(tick)
        parts.append(f'<line x1="{px:.1f}" y1="{y_top}" x2="{px:.1f}" y2="{y_top + plot_height}" stroke="#f2f2f2"/>')
        parts.append(f'<text x="{px:.1f}" y="{y_top + plot_height + 18:.1f}" text-anchor="middle" font-size="11" font-family="sans-serif">{tick}</text>')

    parts.append(f'<text x="{x_left + plot_width / 2:.1f}" y="{y_top + plot_height + 36:.1f}" text-anchor="middle" font-size="12" font-family="sans-serif">Checkpoint iteration</text>')
    parts.append(
        f'<text x="{x0 + 18:.1f}" y="{y_top + plot_height / 2:.1f}" text-anchor="middle" font-size="12" font-family="sans-serif" transform="rotate(-90 {x0 + 18:.1f},{y_top + plot_height / 2:.1f})">{y_label}</text>'
    )

    for label, points, color in series:
        scaled_points = [(sx(x_value), y_value) for x_value, y_value in points]
        for segment in svg_line_segments(scaled_points):
            rendered = [(x_value, sy(y_value)) for x_value, y_value in segment]
            parts.append(f'<polyline fill="none" stroke="{color}" stroke-width="2.2" points="{svg_polyline(rendered)}"/>')
            for px, py in rendered:
                parts.append(f'<circle cx="{px:.2f}" cy="{py:.2f}" r="3" fill="{color}"/>')

    legend_x = x_left + 8
    legend_y = y_top + 16
    for index, (label, _, color) in enumerate(series):
        ly = legend_y + index * 16
        parts.append(f'<line x1="{legend_x}" y1="{ly}" x2="{legend_x + 18}" y2="{ly}" stroke="{color}" stroke-width="2.2"/>')
        parts.append(f'<text x="{legend_x + 24}" y="{ly + 4}" font-size="11" font-family="sans-serif">{label}</text>')

    note_y = y_top + 14
    for note in notes:
        parts.append(f'<text x="{x_left + plot_width - 8:.1f}" y="{note_y:.1f}" text-anchor="end" font-size="11" font-family="sans-serif" fill="#666">{note}</text>')
        note_y += 14


def build_svg(checkpoint_rows: list[dict[str, Any]], output_path: Path) -> None:
    by_run: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in checkpoint_rows:
        by_run[row["run_name"]].append(row)

    run_names = sorted(by_run, key=parse_run_name)
    train_series = []
    eval_loss_series = []
    eval_acc_series = []
    train_notes = []
    eval_notes = []

    for run_name in run_names:
        rows = sorted(by_run[run_name], key=lambda row: row["checkpoint"])
        color = RUN_COLOR_BY_NAME.get(run_name, "#444444")
        label = run_name.replace("_lambda_0.1", " +lambda0.1")
        train_points = [(row["checkpoint"], row["train_mean_loss"]) for row in rows]
        eval_loss_points = [(row["checkpoint"], row["eval_mean_loss"]) for row in rows]
        eval_acc_points = [(row["checkpoint"], row["eval_mean_accuracy"]) for row in rows]
        train_series.append((label, train_points, color))
        eval_loss_series.append((label, eval_loss_points, color))
        eval_acc_series.append((label, eval_acc_points, color))

        if not any(value is not None for _, value in train_points):
            train_notes.append(f"{label}: no recoverable train history")
        if not any(value is not None for _, value in eval_acc_points):
            eval_notes.append(f"{label}: no eval metrics")

    width = 1240
    height = 980
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#faf9f6"/>',
        f'<text x="{width / 2:.1f}" y="28" text-anchor="middle" font-size="22" font-family="sans-serif">RMO-DPO mdga-chi2 checkpoint analysis</text>',
        f'<text x="{width / 2:.1f}" y="52" text-anchor="middle" font-size="13" font-family="sans-serif" fill="#555">source = local .wandb histories + saved eval JSONs under outputs/noise_sweep_mdga_chi2</text>',
    ]

    panel_x = 36
    panel_width = width - 72
    panel_height = 275
    render_panel(
        parts,
        title="Train mean loss by checkpoint",
        series=train_series,
        x0=panel_x,
        y0=80,
        width=panel_width,
        height=panel_height,
        y_label="Train mean loss",
        notes=train_notes,
    )
    render_panel(
        parts,
        title="Eval mean loss by checkpoint",
        series=eval_loss_series,
        x0=panel_x,
        y0=392,
        width=panel_width,
        height=panel_height,
        y_label="Eval mean loss",
        notes=[],
    )
    render_panel(
        parts,
        title="Eval mean accuracy by checkpoint",
        series=eval_acc_series,
        x0=panel_x,
        y0=704,
        width=panel_width,
        height=panel_height,
        y_label="Eval mean accuracy",
        notes=eval_notes,
    )
    parts.append("</svg>")

    ensure_dir(output_path.parent)
    output_path.write_text("\n".join(parts), encoding="utf-8")


def main() -> None:
    args = parse_args()
    repo_root = Path(args.repo_root).resolve()
    sweep_dir = repo_root / args.sweep_dir
    output_dir = repo_root / args.output_dir

    checkpoint_rows, manifest_rows = build_checkpoint_rows(repo_root, sweep_dir)
    summary_rows = summarize_runs(checkpoint_rows)

    checkpoint_csv = output_dir / "mdga_chi2_checkpoint_metrics.csv"
    manifest_csv = output_dir / "mdga_chi2_run_manifest.csv"
    summary_csv = output_dir / "mdga_chi2_run_summary.csv"
    svg_path = output_dir / "mdga_chi2_checkpoint_metrics.svg"
    json_path = output_dir / "mdga_chi2_run_summary.json"

    write_csv(checkpoint_csv, list(checkpoint_rows[0].keys()), checkpoint_rows)
    write_csv(manifest_csv, list(manifest_rows[0].keys()), manifest_rows)
    write_csv(summary_csv, list(summary_rows[0].keys()), summary_rows)
    write_json(json_path, summary_rows)
    build_svg(checkpoint_rows, svg_path)

    print("run_name\tbest_eval_ckpt\tbest_eval_acc\tfinal_ckpt\tfinal_train_loss\tfinal_eval_acc\tfinal_eval_loss\tnotes")
    for row in summary_rows:
        def fmt(value: Any) -> str:
            if isinstance(value, float):
                return f"{value:.6f}"
            return str(value) if value != "" else "NA"

        print(
            "\t".join(
                [
                    row["run_name"],
                    fmt(row["best_eval_checkpoint"]),
                    fmt(row["best_eval_mean_accuracy"]),
                    fmt(row["final_checkpoint"]),
                    fmt(row["final_train_mean_loss"]),
                    fmt(row["final_eval_mean_accuracy"]),
                    fmt(row["final_eval_mean_loss"]),
                    row["notes"] or "OK",
                ]
            )
        )

    print()
    print(f"Checkpoint CSV: {checkpoint_csv}")
    print(f"Manifest CSV:   {manifest_csv}")
    print(f"Summary CSV:    {summary_csv}")
    print(f"Summary JSON:   {json_path}")
    print(f"SVG:            {svg_path}")


if __name__ == "__main__":
    main()
