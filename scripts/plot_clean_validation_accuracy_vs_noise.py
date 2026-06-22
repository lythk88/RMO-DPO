#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


OBJECTIVES = ["helpfulness", "correctness", "coherence", "complexity", "verbosity"]
OBJECTIVE_COLORS = {
    "helpfulness": "#2a9d8f",
    "correctness": "#457b9d",
    "coherence": "#8d5fd3",
    "complexity": "#e76f51",
    "verbosity": "#f4a261",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot clean-validation objective accuracy against noise rate using the final available checkpoint per run."
    )
    parser.add_argument(
        "--repo_root",
        default=str(Path(__file__).resolve().parents[1]),
        help="Repository root that contains the plotting inputs.",
    )
    parser.add_argument(
        "--primary_root",
        default="outputs/noise_sweep",
        help="Primary root directory relative to repo_root containing noise_<rate> subdirectories.",
    )
    parser.add_argument(
        "--noise_rates",
        nargs="+",
        type=float,
        default=[0.0, 0.1, 0.2, 0.3],
        help="Noise rates to include.",
    )
    parser.add_argument(
        "--eval_dir_name",
        default="evals_clean_validation",
        help="Per-noise subdirectory containing evaluation JSON files.",
    )
    parser.add_argument(
        "--output_dir",
        default="outputs/noise_sweep/analysis",
        help="Output directory relative to repo_root.",
    )
    parser.add_argument(
        "--output_stem",
        default="clean_validation_accuracy_vs_noise",
        help="Base filename for generated artifacts.",
    )
    parser.add_argument(
        "--title",
        default="Clean-Validation Accuracy vs Noise Rate",
        help="Figure title.",
    )
    parser.add_argument(
        "--primary_label",
        default="RMO-DPO",
        help="Legend label for the primary series.",
    )
    parser.add_argument(
        "--compare_root",
        default=None,
        help="Optional root directory containing comparison runs grouped by noise-specific subdirectories.",
    )
    parser.add_argument(
        "--compare_label",
        default="RACO",
        help="Legend label for the comparison series.",
    )
    parser.add_argument(
        "--compare_noise_prefix",
        default="helpsteer2_noise_",
        help="Prefix used for comparison-series noise directories under compare_root.",
    )
    parser.add_argument(
        "--compare_run_glob",
        default="raco_*",
        help="Glob used to select the comparison run directory inside each noise directory.",
    )
    parser.add_argument(
        "--compare_eval_dir_name",
        default="evals_helpsteer2",
        help="Evaluation subdirectory name inside the selected comparison run.",
    )
    parser.add_argument(
        "--compare_override",
        action="append",
        default=[],
        metavar="NOISE=DIR",
        help=(
            "Optional per-noise comparison run directory override. "
            "DIR should point at the comparison run directory that contains the compare eval dir."
        ),
    )
    parser.add_argument(
        "--overlay_root",
        default=None,
        help="Optional extra root directory relative to repo_root containing noise_<rate> subdirectories.",
    )
    parser.add_argument(
        "--overlay_label",
        default="RMO-DPO (MDGA-chi2)",
        help="Legend label for the optional extra series.",
    )
    parser.add_argument(
        "--overlay_eval_dir_name",
        default="evals_helpsteer2",
        help="Evaluation subdirectory name inside overlay_root noise directories.",
    )
    parser.add_argument(
        "--overlay_override",
        action="append",
        default=[],
        metavar="NOISE=DIR",
        help=(
            "Optional per-noise overlay run directory override. "
            "DIR should point at the noise run directory that contains the overlay eval dir."
        ),
    )
    parser.add_argument(
        "--extra_root",
        default=None,
        help="Optional second extra root directory relative to repo_root containing noise_<rate> subdirectories.",
    )
    parser.add_argument(
        "--extra_label",
        default="RMO-DPO (lambda 0.1)",
        help="Legend label for the optional fourth series.",
    )
    parser.add_argument(
        "--extra_eval_dir_name",
        default="evals_helpsteer2",
        help="Evaluation subdirectory name inside extra_root noise directories.",
    )
    parser.add_argument(
        "--extra_override",
        action="append",
        default=[],
        metavar="NOISE=DIR",
        help=(
            "Optional per-noise fourth-series run directory override. "
            "DIR should point at the noise run directory that contains the extra eval dir."
        ),
    )
    return parser.parse_args()


def noise_tag(noise_rate: float) -> str:
    return f"{noise_rate:.1f}"


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def display_path(path: Path, *roots: Path) -> str:
    for root in roots:
        try:
            return str(path.relative_to(root))
        except ValueError:
            continue
    return str(path)


def checkpoint_step(path: Path) -> int:
    return int(path.stem.split("-", 1)[1])


def select_final_available_eval(eval_dir: Path) -> Path:
    final_json = eval_dir / "final.json"
    if final_json.exists():
        return final_json
    checkpoint_jsons = sorted(eval_dir.glob("checkpoint-*.json"), key=checkpoint_step)
    if checkpoint_jsons:
        return checkpoint_jsons[-1]
    raise FileNotFoundError(f"No evaluation JSON files found in {eval_dir}")


def select_final_available_eval_optional(eval_dir: Path) -> Path | None:
    if not eval_dir.exists():
        return None
    final_json = eval_dir / "final.json"
    if final_json.exists():
        return final_json
    checkpoint_jsons = sorted(eval_dir.glob("checkpoint-*.json"), key=checkpoint_step)
    return checkpoint_jsons[-1] if checkpoint_jsons else None


def parse_noise_dir_overrides(overrides: list[str], repo_root: Path) -> dict[str, Path]:
    parsed: dict[str, Path] = {}
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"Invalid override '{item}'. Expected NOISE=DIR.")
        noise_value, dir_value = item.split("=", 1)
        tag = noise_tag(float(noise_value))
        run_dir = Path(dir_value)
        if not run_dir.is_absolute():
            run_dir = (repo_root / run_dir).resolve()
        parsed[tag] = run_dir
    return parsed


def polyline(points: list[tuple[float, float]]) -> str:
    return " ".join(f"{x:.2f},{y:.2f}" for x, y in points)


def noise_matches_dir(name: str, prefix: str, noise_rate: float) -> bool:
    if not name.startswith(prefix):
        return False
    suffix = name[len(prefix) :]
    try:
        return math.isclose(float(suffix), float(noise_rate), abs_tol=1e-9)
    except ValueError:
        return False


def select_compare_eval_dir(
    compare_root: Path,
    noise_rate: float,
    *,
    noise_prefix: str,
    run_glob: str,
    eval_dir_name: str,
) -> tuple[Path, Path]:
    noise_dirs = sorted(
        (path for path in compare_root.iterdir() if path.is_dir() and noise_matches_dir(path.name, noise_prefix, noise_rate)),
        key=lambda path: path.stat().st_mtime,
    )
    if not noise_dirs:
        raise FileNotFoundError(f"No comparison noise directory found for noise rate {noise_rate:.1f} under {compare_root}")
    noise_dir = noise_dirs[-1]

    run_dirs = sorted((path for path in noise_dir.glob(run_glob) if path.is_dir()), key=lambda path: path.stat().st_mtime)
    if not run_dirs:
        run_dirs = sorted((path for path in noise_dir.iterdir() if path.is_dir()), key=lambda path: path.stat().st_mtime)
    if not run_dirs:
        raise FileNotFoundError(f"No comparison run directories found under {noise_dir}")
    run_dir = run_dirs[-1]

    eval_dir = run_dir / eval_dir_name
    if not eval_dir.exists():
        raise FileNotFoundError(f"Comparison eval directory not found: {eval_dir}")
    return run_dir, eval_dir


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


def is_finite_number(value: Any) -> bool:
    return isinstance(value, int | float) and math.isfinite(float(value))


def format_optional_accuracy(value: Any) -> str:
    return f"{float(value):.6f}" if is_finite_number(value) else ""


def finite_point_runs(xs: list[float], ys: list[Any]) -> list[list[tuple[float, float]]]:
    runs: list[list[tuple[float, float]]] = []
    current: list[tuple[float, float]] = []
    for x_value, y_value in zip(xs, ys, strict=True):
        if is_finite_number(y_value):
            current.append((x_value, float(y_value)))
        elif current:
            runs.append(current)
            current = []
    if current:
        runs.append(current)
    return runs


def draw_marker(parts: list[str], px: float, py: float, color: str, marker: str) -> None:
    if marker == "filled-circle":
        parts.append(f'<circle cx="{px:.2f}" cy="{py:.2f}" r="4" fill="{color}"/>')
        return
    if marker == "open-circle":
        parts.append(f'<circle cx="{px:.2f}" cy="{py:.2f}" r="4" fill="white" stroke="{color}" stroke-width="2"/>')
        return
    if marker == "open-square":
        parts.append(
            f'<rect x="{px - 3.5:.2f}" y="{py - 3.5:.2f}" width="7" height="7" fill="white" stroke="{color}" stroke-width="2"/>'
        )
        return
    if marker == "open-diamond":
        parts.append(
            f'<polygon points="{px:.2f},{py - 4.5:.2f} {px + 4.5:.2f},{py:.2f} {px:.2f},{py + 4.5:.2f} {px - 4.5:.2f},{py:.2f}" fill="white" stroke="{color}" stroke-width="2"/>'
        )
        return
    raise ValueError(f"Unsupported marker style: {marker}")


def draw_series(
    parts: list[str],
    xs: list[float],
    ys: list[Any],
    color: str,
    *,
    x0: float,
    y0: float,
    w: float,
    h: float,
    xmin: float,
    xmax: float,
    ymin: float,
    ymax: float,
    stroke_width: float,
    dasharray: str | None,
    marker: str,
    label_points: bool,
) -> None:
    runs = finite_point_runs(xs, ys)
    for run in runs:
        run_xs = [x_value for x_value, _ in run]
        run_ys = [y_value for _, y_value in run]
        points = scale_points(run_xs, run_ys, x0, y0, w, h, xmin, xmax, ymin, ymax)
        if len(points) >= 2:
            dash_attr = f' stroke-dasharray="{dasharray}"' if dasharray else ""
            parts.append(
                f'<polyline fill="none" stroke="{color}" stroke-width="{stroke_width}"{dash_attr} opacity="0.95" points="{polyline(points)}"/>'
            )
        for (px, py), value in zip(points, run_ys, strict=True):
            draw_marker(parts, px, py, color, marker)
            if label_points:
                parts.append(
                    f'<text x="{px:.2f}" y="{py - 10:.2f}" text-anchor="middle" font-size="10" font-family="sans-serif" fill="{color}">{value:.3f}</text>'
                )


def draw_panel(
    parts: list[str],
    title: str,
    xs: list[float],
    ys: list[float],
    color: str,
    *,
    compare_ys: list[float] | None = None,
    overlay_ys: list[float] | None = None,
    extra_ys: list[float | None] | None = None,
    x0: float,
    y0: float,
    w: float,
    h: float,
    xmin: float,
    xmax: float,
    ymin: float,
    ymax: float,
) -> None:
    parts.append(f'<rect x="{x0}" y="{y0}" width="{w}" height="{h}" fill="white" stroke="#d7d7d7"/>')
    parts.append(
        f'<text x="{x0 + w / 2:.1f}" y="{y0 - 12:.1f}" text-anchor="middle" font-size="18" font-family="sans-serif">{title}</text>'
    )

    for frac in [0.0, 0.25, 0.5, 0.75, 1.0]:
        py = y0 + h - frac * h
        value = ymin + frac * (ymax - ymin)
        parts.append(f'<line x1="{x0}" y1="{py:.1f}" x2="{x0 + w}" y2="{py:.1f}" stroke="#ededed"/>')
        parts.append(
            f'<text x="{x0 - 8:.1f}" y="{py + 4:.1f}" text-anchor="end" font-size="11" font-family="sans-serif">{value:.3f}</text>'
        )

    for tick in xs:
        px = x0 + ((tick - xmin) / (xmax - xmin) * w if xmax > xmin else 0.5 * w)
        parts.append(f'<line x1="{px:.1f}" y1="{y0}" x2="{px:.1f}" y2="{y0 + h}" stroke="#f5f5f5"/>')
        parts.append(
            f'<text x="{px:.1f}" y="{y0 + h + 18:.1f}" text-anchor="middle" font-size="11" font-family="sans-serif">{tick:.1f}</text>'
        )

    parts.append(
        f'<text x="{x0 + w / 2:.1f}" y="{y0 + h + 38:.1f}" text-anchor="middle" font-size="12" font-family="sans-serif">Noise rate</text>'
    )
    parts.append(
        f'<text x="{x0 - 44:.1f}" y="{y0 + h / 2:.1f}" text-anchor="middle" font-size="12" font-family="sans-serif" transform="rotate(-90 {x0 - 44:.1f},{y0 + h / 2:.1f})">Accuracy</text>'
    )

    if compare_ys is not None:
        draw_series(
            parts,
            xs,
            compare_ys,
            color,
            x0=x0,
            y0=y0,
            w=w,
            h=h,
            xmin=xmin,
            xmax=xmax,
            ymin=ymin,
            ymax=ymax,
            stroke_width=2.5,
            dasharray="10 6",
            marker="open-square",
            label_points=False,
        )

    if overlay_ys is not None:
        draw_series(
            parts,
            xs,
            overlay_ys,
            color,
            x0=x0,
            y0=y0,
            w=w,
            h=h,
            xmin=xmin,
            xmax=xmax,
            ymin=ymin,
            ymax=ymax,
            stroke_width=2.5,
            dasharray="3 5",
            marker="open-circle",
            label_points=False,
        )

    if extra_ys is not None:
        draw_series(
            parts,
            xs,
            extra_ys,
            color,
            x0=x0,
            y0=y0,
            w=w,
            h=h,
            xmin=xmin,
            xmax=xmax,
            ymin=ymin,
            ymax=ymax,
            stroke_width=2.5,
            dasharray="12 4 3 4",
            marker="open-diamond",
            label_points=False,
        )

    draw_series(
        parts,
        xs,
        ys,
        color,
        x0=x0,
        y0=y0,
        w=w,
        h=h,
        xmin=xmin,
        xmax=xmax,
        ymin=ymin,
        ymax=ymax,
        stroke_width=3.0,
        dasharray=None,
        marker="filled-circle",
        label_points=True,
    )


def resolve_optional_root(root_value: str | None, repo_root: Path) -> Path | None:
    if not root_value:
        return None
    root_path = Path(root_value)
    if not root_path.is_absolute():
        root_path = (repo_root / root_path).resolve()
    return root_path


def set_optional_series_row_missing(row: dict[str, Any], prefix: str) -> None:
    row[f"{prefix}_run_dir"] = None
    row[f"{prefix}_selected_file"] = None
    row[f"{prefix}_selected_label"] = None
    row[f"{prefix}_checkpoint"] = None
    row[f"{prefix}_mean_accuracy"] = None
    for objective in OBJECTIVES:
        row[f"{prefix}_{objective}/accuracy"] = None


def maybe_add_optional_series(
    row: dict[str, Any],
    *,
    prefix: str,
    root: Path | None,
    overrides: dict[str, Path],
    tag: str,
    repo_root: Path,
    display_root: Path | None,
    eval_dir_name: str,
) -> None:
    run_dir = overrides.get(tag)
    if run_dir is None and root is not None:
        run_dir = root / f"noise_{tag}"
    if run_dir is None:
        set_optional_series_row_missing(row, prefix)
        return
    eval_json = select_final_available_eval_optional(run_dir / eval_dir_name)
    if eval_json is None:
        set_optional_series_row_missing(row, prefix)
        return

    payload = json.loads(eval_json.read_text(encoding="utf-8"))
    metrics = payload["metrics"]
    roots = [repo_root]
    if display_root is not None:
        roots.append(display_root)
    row[f"{prefix}_run_dir"] = display_path(run_dir, *roots)
    row[f"{prefix}_selected_file"] = display_path(eval_json, *roots)
    row[f"{prefix}_selected_label"] = eval_json.stem
    row[f"{prefix}_checkpoint"] = payload.get("checkpoint")
    row[f"{prefix}_mean_accuracy"] = metrics["mean_accuracy"]
    for objective in OBJECTIVES:
        row[f"{prefix}_{objective}/accuracy"] = metrics[f"{objective}/accuracy"]


def main() -> None:
    args = parse_args()
    repo_root = Path(args.repo_root).resolve()
    primary_root = Path(args.primary_root)
    if not primary_root.is_absolute():
        primary_root = (repo_root / primary_root).resolve()
    overlay_root = resolve_optional_root(args.overlay_root, repo_root)
    extra_root = resolve_optional_root(args.extra_root, repo_root)
    output_dir = (repo_root / args.output_dir).resolve()
    ensure_dir(output_dir)
    compare_overrides = parse_noise_dir_overrides(args.compare_override, repo_root)
    overlay_overrides = parse_noise_dir_overrides(args.overlay_override, repo_root)
    extra_overrides = parse_noise_dir_overrides(args.extra_override, repo_root)

    selected_rows: list[dict[str, Any]] = []
    compare_root = Path(args.compare_root).resolve() if args.compare_root else None
    for noise_rate in args.noise_rates:
        tag = noise_tag(noise_rate)
        eval_dir = primary_root / f"noise_{tag}" / args.eval_dir_name
        chosen_json = select_final_available_eval(eval_dir)
        payload = json.loads(chosen_json.read_text(encoding="utf-8"))
        metrics = payload["metrics"]
        row = {
            "noise_rate": noise_rate,
            "noise_tag": tag,
            "selected_file": display_path(chosen_json, repo_root, primary_root),
            "selected_label": chosen_json.stem,
            "checkpoint": payload.get("checkpoint"),
            "mean_accuracy": metrics["mean_accuracy"],
        }
        for objective in OBJECTIVES:
            row[f"{objective}/accuracy"] = metrics[f"{objective}/accuracy"]

        if compare_root is not None:
            compare_run_dir = compare_overrides.get(tag)
            if compare_run_dir is None:
                compare_run_dir, compare_eval_dir = select_compare_eval_dir(
                    compare_root,
                    noise_rate,
                    noise_prefix=args.compare_noise_prefix,
                    run_glob=args.compare_run_glob,
                    eval_dir_name=args.compare_eval_dir_name,
                )
            else:
                compare_eval_dir = compare_run_dir / args.compare_eval_dir_name
            compare_json = select_final_available_eval(compare_eval_dir)
            compare_payload = json.loads(compare_json.read_text(encoding="utf-8"))
            compare_metrics = compare_payload["metrics"]
            row["compare_run_dir"] = display_path(compare_run_dir, compare_root)
            row["compare_selected_file"] = display_path(compare_json, compare_root)
            row["compare_selected_label"] = compare_json.stem
            row["compare_checkpoint"] = compare_payload.get("checkpoint")
            row["compare_mean_accuracy"] = compare_metrics["mean_accuracy"]
            for objective in OBJECTIVES:
                row[f"compare_{objective}/accuracy"] = compare_metrics[f"{objective}/accuracy"]

        if overlay_root is not None:
            maybe_add_optional_series(
                row,
                prefix="overlay",
                root=overlay_root,
                overrides=overlay_overrides,
                tag=tag,
                repo_root=repo_root,
                display_root=overlay_root,
                eval_dir_name=args.overlay_eval_dir_name,
            )
        if extra_root is not None or extra_overrides:
            maybe_add_optional_series(
                row,
                prefix="extra",
                root=extra_root,
                overrides=extra_overrides,
                tag=tag,
                repo_root=repo_root,
                display_root=extra_root,
                eval_dir_name=args.extra_eval_dir_name,
            )
        selected_rows.append(row)

    json_path = output_dir / f"{args.output_stem}.json"
    json_path.write_text(json.dumps(selected_rows, indent=2), encoding="utf-8")

    tsv_lines = [
        "\t".join(
            [
                "noise_rate",
                "selected_label",
                "selected_file",
                "checkpoint",
                "mean_accuracy",
                *[f"{objective}_accuracy" for objective in OBJECTIVES],
                *(
                    [
                        "compare_run_dir",
                        "compare_selected_label",
                        "compare_selected_file",
                        "compare_checkpoint",
                        "compare_mean_accuracy",
                        *[f"compare_{objective}_accuracy" for objective in OBJECTIVES],
                    ]
                    if compare_root is not None
                    else []
                ),
                *(
                    [
                        "overlay_run_dir",
                        "overlay_selected_label",
                        "overlay_selected_file",
                        "overlay_checkpoint",
                        "overlay_mean_accuracy",
                        *[f"overlay_{objective}_accuracy" for objective in OBJECTIVES],
                    ]
                    if overlay_root is not None
                    else []
                ),
                *(
                    [
                        "extra_run_dir",
                        "extra_selected_label",
                        "extra_selected_file",
                        "extra_checkpoint",
                        "extra_mean_accuracy",
                        *[f"extra_{objective}_accuracy" for objective in OBJECTIVES],
                    ]
                    if extra_root is not None or extra_overrides
                    else []
                ),
            ]
        )
    ]
    for row in selected_rows:
        tsv_lines.append(
            "\t".join(
                [
                    f"{row['noise_rate']:.1f}",
                    row["selected_label"],
                    row["selected_file"],
                    str(row["checkpoint"]),
                    f"{row['mean_accuracy']:.6f}",
                    *[f"{row[f'{objective}/accuracy']:.6f}" for objective in OBJECTIVES],
                    *(
                        [
                            str(row["compare_run_dir"]),
                            row["compare_selected_label"],
                            row["compare_selected_file"],
                            str(row["compare_checkpoint"]),
                            f"{row['compare_mean_accuracy']:.6f}",
                            *[f"{row[f'compare_{objective}/accuracy']:.6f}" for objective in OBJECTIVES],
                        ]
                        if compare_root is not None
                        else []
                    ),
                    *(
                        [
                            str(row["overlay_run_dir"]),
                            row["overlay_selected_label"],
                            row["overlay_selected_file"],
                            str(row["overlay_checkpoint"]),
                            f"{row['overlay_mean_accuracy']:.6f}",
                            *[f"{row[f'overlay_{objective}/accuracy']:.6f}" for objective in OBJECTIVES],
                        ]
                        if overlay_root is not None
                        else []
                    ),
                    *(
                        [
                            str(row["extra_run_dir"] or ""),
                            str(row["extra_selected_label"] or ""),
                            str(row["extra_selected_file"] or ""),
                            str(row["extra_checkpoint"] or ""),
                            format_optional_accuracy(row["extra_mean_accuracy"]),
                            *[format_optional_accuracy(row[f"extra_{objective}/accuracy"]) for objective in OBJECTIVES],
                        ]
                        if extra_root is not None or extra_overrides
                        else []
                    ),
                ]
            )
        )
    tsv_path = output_dir / f"{args.output_stem}.tsv"
    tsv_path.write_text("\n".join(tsv_lines) + "\n", encoding="utf-8")

    xs = [row["noise_rate"] for row in selected_rows]
    all_accuracies = [
        row[f"{objective}/accuracy"] for row in selected_rows for objective in OBJECTIVES
    ]
    if compare_root is not None:
        all_accuracies.extend(
            row[f"compare_{objective}/accuracy"] for row in selected_rows for objective in OBJECTIVES
        )
    if overlay_root is not None:
        all_accuracies.extend(
            row[f"overlay_{objective}/accuracy"]
            for row in selected_rows
            for objective in OBJECTIVES
            if is_finite_number(row[f"overlay_{objective}/accuracy"])
        )
    if extra_root is not None or extra_overrides:
        all_accuracies.extend(
            row[f"extra_{objective}/accuracy"]
            for row in selected_rows
            for objective in OBJECTIVES
            if is_finite_number(row[f"extra_{objective}/accuracy"])
        )
    ymin = min(all_accuracies)
    ymax = max(all_accuracies)
    if math.isclose(ymin, ymax):
        ymax = ymin + 1.0
    pad = (ymax - ymin) * 0.10
    ymin -= pad
    ymax += pad

    has_extra_series = extra_root is not None or bool(extra_overrides)
    width = 1400
    height = 980 if has_extra_series else 920
    panel_w = 340
    panel_h = 240
    top_y = 200 if has_extra_series else 170
    bottom_y = 560 if has_extra_series else 530
    top_xs = [80, 530, 980]
    bottom_xs = [305, 755]
    positions = [
        (top_xs[0], top_y),
        (top_xs[1], top_y),
        (top_xs[2], top_y),
        (bottom_xs[0], bottom_y),
        (bottom_xs[1], bottom_y),
    ]

    selection_summary = ", ".join(
        f"{row['noise_tag']} -> {row['selected_label']}" for row in selected_rows
    )
    compare_summary = None
    if compare_root is not None:
        compare_summary = ", ".join(
            f"{row['noise_tag']} -> {row['compare_run_dir']}/{row['compare_selected_label']}" for row in selected_rows
        )
    overlay_summary = None
    if overlay_root is not None:
        overlay_summary = ", ".join(
            f"{row['noise_tag']} -> {Path(str(row['overlay_run_dir'])).name}/{row['overlay_selected_label']}"
            for row in selected_rows
        )
    extra_summary = None
    if has_extra_series:
        extra_entries = [
            f"{row['noise_tag']} -> {Path(str(row['extra_run_dir'])).name}/{row['extra_selected_label']}"
            for row in selected_rows
            if row["extra_run_dir"] and row["extra_selected_label"]
        ]
        extra_summary = ", ".join(extra_entries) if extra_entries else "no evaluation artifacts found"

    legend_fragments = [f"Solid: {args.primary_label}"]
    if compare_root is not None:
        legend_fragments.append(f"dashed: {args.compare_label}")
    if overlay_root is not None:
        legend_fragments.append(f"dotted: {args.overlay_label}")
    if has_extra_series:
        legend_fragments.append(f"dash-dot: {args.extra_label}")

    summary_lines = [
        f"{args.primary_label} artifacts: {selection_summary}",
    ]
    if compare_summary is not None:
        summary_lines.append(f"{args.compare_label} artifacts: {compare_summary}")
    if overlay_summary is not None:
        summary_lines.append(f"{args.overlay_label} artifacts: {overlay_summary}")
    if extra_summary is not None:
        summary_lines.append(f"{args.extra_label} artifacts: {extra_summary}")

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#faf9f6"/>',
        f'<text x="{width / 2:.1f}" y="34" text-anchor="middle" font-size="24" font-family="sans-serif">{args.title}</text>',
        '<text x="700" y="60" text-anchor="middle" font-size="13" font-family="sans-serif" fill="#444444">Each panel uses final.json if present, otherwise the latest available checkpoint JSON for that noise rate.</text>',
        f'<text x="700" y="82" text-anchor="middle" font-size="13" font-family="sans-serif" fill="#444444">{"; ".join(legend_fragments)}.</text>',
    ]
    for index, line in enumerate(summary_lines, start=0):
        y_value = 104 + index * 22
        parts.append(
            f'<text x="700" y="{y_value}" text-anchor="middle" font-size="13" font-family="sans-serif" fill="#444444">{line}</text>'
        )

    for objective, (x0, y0) in zip(OBJECTIVES, positions, strict=True):
        ys = [row[f"{objective}/accuracy"] for row in selected_rows]
        compare_ys = [row[f"compare_{objective}/accuracy"] for row in selected_rows] if compare_root is not None else None
        overlay_ys = [row[f"overlay_{objective}/accuracy"] for row in selected_rows] if overlay_root is not None else None
        extra_ys = [row[f"extra_{objective}/accuracy"] for row in selected_rows] if has_extra_series else None
        draw_panel(
            parts,
            objective.capitalize(),
            xs,
            ys,
            OBJECTIVE_COLORS[objective],
            compare_ys=compare_ys,
            overlay_ys=overlay_ys,
            extra_ys=extra_ys,
            x0=x0,
            y0=y0,
            w=panel_w,
            h=panel_h,
            xmin=min(xs),
            xmax=max(xs),
            ymin=ymin,
            ymax=ymax,
        )

    parts.append("</svg>")
    svg_path = output_dir / f"{args.output_stem}.svg"
    svg_path.write_text("\n".join(parts), encoding="utf-8")

    print(f"Wrote {svg_path}")
    print(f"Wrote {tsv_path}")
    print(f"Wrote {json_path}")


if __name__ == "__main__":
    main()
