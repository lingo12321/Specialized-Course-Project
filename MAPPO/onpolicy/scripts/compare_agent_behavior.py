#!/usr/bin/env python
"""Compare MAPPO behavior-evaluation CSV files and create figures/report."""

from __future__ import annotations

import argparse
import csv
import math
import time
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


EPISODE_REQUIRED = {
    "model_name", "episode", "episode_reward", "avg_coverage",
    "final_coverage", "hard_success", "avg_soft_coverage",
    "final_soft_coverage", "soft_success", "avg_distance_coverage",
    "final_distance_coverage", "duplicate_step_rate", "collision_count",
}
TIMESTEP_REQUIRED = {
    "model_name", "episode", "timestep", "coverage", "soft_coverage",
    "distance_coverage", "duplicate_rate", "collision_pair_count",
}
BAR_SPECS = (
    ("episode_reward", "episode_reward.png", "Episode Reward", "Higher is Better"),
    ("avg_coverage", "average_coverage.png", "Average Coverage", "Higher is Better"),
    ("final_coverage", "final_coverage.png", "Final Coverage", "Higher is Better"),
    (
        "avg_soft_coverage", "soft_coverage.png",
        "Average Soft Coverage", "Higher is Better",
    ),
    (
        "avg_distance_coverage", "distance_coverage.png",
        "Average Distance-based Coverage", "Higher is Better",
    ),
    ("hard_success", "success_rate.png", "Hard Success Rate", "Higher is Better"),
    (
        "duplicate_step_rate", "duplicate_step_rate.png",
        "Duplicate Step Rate", "Lower is Better",
    ),
    ("collision_count", "collision_count.png", "Collision Count", "Lower is Better"),
)
CURVE_SPECS = (
    ("coverage", "coverage_over_time.png", "Coverage over Time", "Coverage"),
    (
        "duplicate_rate", "duplicate_over_time.png",
        "Duplicate Occupancy over Time", "Duplicate Rate",
    ),
    (
        "collision_pair_count", "collision_over_time.png",
        "Collisions over Time", "Collision Pair Count",
    ),
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse comparison CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--csv_files", type=Path, nargs="+", required=True,
        help="Two or more episode_metrics.csv files.",
    )
    parser.add_argument("--output_dir", type=Path, required=True)
    args = parser.parse_args(argv)
    if len(args.csv_files) < 2:
        parser.error("--csv_files requires at least two episode_metrics.csv files.")
    return args


def prepare_output_dir(requested: Path) -> Path:
    """Create a new result directory without overwriting existing files."""
    requested = requested.expanduser().resolve()
    if requested.exists() and any(requested.iterdir()):
        requested = requested.with_name(
            f"{requested.name}_{time.strftime('%Y%m%d_%H%M%S')}"
        )
        print(f"Output directory is not empty; writing to: {requested}")
    requested.mkdir(parents=True, exist_ok=True)
    return requested


def read_csv(path: Path, required: Iterable[str]) -> List[Dict[str, str]]:
    """Read and validate a nonempty CSV."""
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"CSV file does not exist: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = set(reader.fieldnames or [])
        missing = set(required) - fields
        if missing:
            raise ValueError(
                f"CSV is missing required fields {sorted(missing)}: {path}"
            )
        rows = list(reader)
    if not rows:
        raise ValueError(f"CSV contains no data rows: {path}")
    return rows


def numeric(rows: Sequence[Mapping[str, str]], field: str) -> np.ndarray:
    """Convert one CSV column to finite floats with contextual errors."""
    try:
        values = np.asarray([float(row[field]) for row in rows], dtype=float)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Field {field!r} contains nonnumeric data.") from exc
    if not np.all(np.isfinite(values)):
        raise ValueError(f"Field {field!r} contains non-finite values.")
    return values


def sample_std(values: np.ndarray) -> float:
    """Return sample standard deviation, or zero for a single observation."""
    return float(np.std(values, ddof=1)) if len(values) > 1 else 0.0


def load_inputs(
    paths: Sequence[Path],
) -> tuple[Dict[str, List[Dict[str, str]]], Dict[str, List[Dict[str, str]]]]:
    """Load episode CSVs and their sibling timestep_metrics.csv files."""
    episodes: Dict[str, List[Dict[str, str]]] = {}
    timesteps: Dict[str, List[Dict[str, str]]] = {}
    for path in paths:
        episode_rows = read_csv(path, EPISODE_REQUIRED)
        names = {row["model_name"].strip() for row in episode_rows}
        if len(names) != 1 or "" in names:
            raise ValueError(f"Expected exactly one nonempty model_name in {path}.")
        name = next(iter(names))
        if name in episodes:
            raise ValueError(f"Duplicate model_name across inputs: {name}")
        timestep_path = path.expanduser().resolve().parent / "timestep_metrics.csv"
        timestep_rows = read_csv(timestep_path, TIMESTEP_REQUIRED)
        timestep_names = {row["model_name"].strip() for row in timestep_rows}
        if timestep_names != {name}:
            raise ValueError(
                f"Model names do not match between {path} and {timestep_path}."
            )
        episodes[name] = episode_rows
        timesteps[name] = timestep_rows
    return episodes, timesteps


def summaries(
    episodes: Mapping[str, Sequence[Mapping[str, str]]],
) -> List[Dict[str, float | str | int]]:
    """Build one comparison summary row per model."""
    rows: List[Dict[str, float | str | int]] = []
    for name, model_rows in episodes.items():
        row: Dict[str, float | str | int] = {
            "model_name": name,
            "num_episodes": len(model_rows),
        }
        for field, _, _, _ in BAR_SPECS:
            values = numeric(model_rows, field)
            label = "hard_success_rate" if field == "hard_success" else field
            row[f"{label}_mean"] = float(np.mean(values))
            row[f"{label}_std"] = sample_std(values)
        for optional in (
            "final_soft_coverage", "soft_success",
            "final_distance_coverage", "duplicate_rate",
            "duplicate_event_count", "collision_step_rate", "full_coverage_step",
        ):
            if optional in model_rows[0]:
                values = numeric(model_rows, optional)
                row[f"{optional}_mean"] = float(np.mean(values))
                row[f"{optional}_std"] = sample_std(values)
        rows.append(row)
    return rows


def save_bar_plots(
    episodes: Mapping[str, Sequence[Mapping[str, str]]], output_dir: Path
) -> None:
    """Save mean ± sample-standard-deviation bar plots."""
    names = list(episodes)
    colors = plt.cm.Set2(np.linspace(0, 1, len(names)))
    for field, filename, title, direction in BAR_SPECS:
        arrays = [numeric(episodes[name], field) for name in names]
        means = [float(np.mean(values)) for values in arrays]
        stds = [sample_std(values) for values in arrays]
        fig, axis = plt.subplots(figsize=(7.5, 5))
        axis.bar(names, means, yerr=stds, capsize=5, color=colors)
        axis.set_title(f"{title} ({direction})")
        axis.set_ylabel(title)
        axis.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        fig.savefig(output_dir / filename, dpi=300)
        plt.close(fig)


def curve_by_timestep(
    rows: Sequence[Mapping[str, str]], metric: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Aggregate a timestep metric across episodes as mean ± sample std."""
    grouped: Dict[int, List[float]] = {}
    for row in rows:
        timestep = int(row["timestep"])
        grouped.setdefault(timestep, []).append(float(row[metric]))
    x = np.asarray(sorted(grouped), dtype=int)
    mean = np.asarray([np.mean(grouped[t]) for t in x], dtype=float)
    std = np.asarray([
        np.std(grouped[t], ddof=1) if len(grouped[t]) > 1 else 0.0 for t in x
    ])
    return x, mean, std


def save_curve_plots(
    timesteps: Mapping[str, Sequence[Mapping[str, str]]], output_dir: Path
) -> None:
    """Save per-timestep mean curves with sample-standard-deviation bands."""
    for metric, filename, title, ylabel in CURVE_SPECS:
        fig, axis = plt.subplots(figsize=(8, 5))
        for name, rows in timesteps.items():
            x, mean, std = curve_by_timestep(rows, metric)
            line = axis.plot(x, mean, label=name, linewidth=2)[0]
            axis.fill_between(
                x, mean - std, mean + std, color=line.get_color(), alpha=0.18
            )
        axis.set_title(f"{title} (Mean ± SD)")
        axis.set_xlabel("Timestep")
        axis.set_ylabel(ylabel)
        axis.grid(alpha=0.25)
        axis.legend()
        fig.tight_layout()
        fig.savefig(output_dir / filename, dpi=300)
        plt.close(fig)


def relation_sentence(
    baseline: Mapping[str, float | str | int],
    candidate: Mapping[str, float | str | int],
    field: str,
    label: str,
) -> str:
    """Describe an observed numerical difference without causal language."""
    left = float(baseline[f"{field}_mean"])
    right = float(candidate[f"{field}_mean"])
    tolerance = 1e-12
    if math.isclose(left, right, abs_tol=tolerance, rel_tol=tolerance):
        relation = "was unchanged"
    elif right < left:
        relation = "was lower"
    else:
        relation = "was higher"
    return (
        f"- {label}: the second model {relation} "
        f"({right:.6f} vs. {left:.6f})."
    )


def write_report(
    summary_rows: Sequence[Mapping[str, float | str | int]], output_dir: Path
) -> None:
    """Write a concise, correlation-only Markdown analysis."""
    first, second = summary_rows[0], summary_rows[1]
    lines = [
        "# Agent Behavior Comparison",
        "",
        "This report describes observed associations in the evaluation data. "
        "It does not establish that the penalty caused any difference.",
        "",
        f"Reference model: **{first['model_name']}**  ",
        f"Second model: **{second['model_name']}**",
        "",
        "## Observed differences",
        "",
        relation_sentence(first, second, "episode_reward", "Episode reward"),
        relation_sentence(first, second, "avg_coverage", "Average coverage"),
        relation_sentence(first, second, "final_coverage", "Final coverage"),
        relation_sentence(
            first, second, "avg_soft_coverage", "Average soft coverage"
        ),
        relation_sentence(
            first, second, "final_soft_coverage", "Final soft coverage"
        ),
        relation_sentence(
            first, second, "avg_distance_coverage",
            "Average distance-based coverage"
        ),
        relation_sentence(
            first, second, "final_distance_coverage",
            "Final distance-based coverage"
        ),
        relation_sentence(
            first, second, "hard_success_rate", "Hard success rate"
        ),
        relation_sentence(
            first, second, "duplicate_step_rate", "Duplicate step rate"
        ),
        relation_sentence(first, second, "collision_count", "Collision count"),
        "",
        (
            "The second model's average soft coverage was "
            f"{float(second['avg_soft_coverage_mean']):.6f}, compared with "
            f"{float(first['avg_soft_coverage_mean']):.6f} for the reference "
            "model. A higher value indicates greater target proximity under the "
            "linear distance score; this descriptive comparison does not establish "
            "that the penalty caused the difference."
        ),
        "",
        "The plots use mean ± sample standard deviation. Results should be "
        "interpreted with the episode count, seed coverage, and occupancy "
        "definition used by the evaluator in mind.",
    ]

    duplicate_change = (
        float(second["duplicate_step_rate_mean"])
        - float(first["duplicate_step_rate_mean"])
    )
    coverage_change = (
        float(second["avg_coverage_mean"]) - float(first["avg_coverage_mean"])
    )
    success_change = (
        float(second["hard_success_rate_mean"])
        - float(first["hard_success_rate_mean"])
    )
    lines.extend(["", "## Suggested next step", ""])
    if duplicate_change >= -1e-12:
        lines.append(
            "Duplicate occupancy did not show a clear numerical reduction. "
            "Before tuning lambda, inspect the penalty trigger, reward integration, "
            "penalized entities, and distance threshold."
        )
    elif coverage_change < -1e-12 or success_change < -1e-12:
        lines.append(
            "Duplicate occupancy was numerically lower while coverage or success "
            "was also lower. A subsequent experiment could evaluate lambda=0.02 "
            "and lambda=0.05; those models were not trained in this evaluation."
        )
    else:
        lines.append(
            "Duplicate occupancy was numerically lower while coverage and success "
            "were broadly maintained. Smaller lambda values may be evaluated next "
            "to examine whether episode reward can be recovered."
        )
    (output_dir / "analysis_report.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def write_summary_csv(
    rows: Sequence[Mapping[str, float | str | int]], path: Path
) -> None:
    """Write the cross-model summary with a union of available fields."""
    fields: List[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main(argv: Sequence[str] | None = None) -> None:
    """CLI entry point."""
    args = parse_args(argv)
    output_dir = prepare_output_dir(args.output_dir)
    episodes, timesteps = load_inputs(args.csv_files)
    summary_rows = summaries(episodes)
    write_summary_csv(summary_rows, output_dir / "comparison_summary.csv")
    save_bar_plots(episodes, output_dir)
    save_curve_plots(timesteps, output_dir)
    write_report(summary_rows, output_dir)
    print(f"Compared models: {', '.join(episodes)}")
    print(f"Results: {output_dir}")


if __name__ == "__main__":
    main()
