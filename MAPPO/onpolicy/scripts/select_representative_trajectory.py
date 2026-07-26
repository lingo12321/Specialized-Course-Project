#!/usr/bin/env python
"""Rank representative paired episodes and render the top trajectory seeds."""

from __future__ import annotations

import argparse
import csv
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Mapping, Sequence

import numpy as np


REQUIRED_FIELDS = {
    "episode", "episode_reward", "duplicate_rate", "avg_distance_coverage",
}
OUTPUT_FIELDS = [
    "seed", "baseline_reward", "penalty_reward", "baseline_duplicate",
    "penalty_duplicate", "baseline_distance_coverage",
    "penalty_distance_coverage", "score",
]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse selection, checkpoint, and rendering paths."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline_csv", type=Path, required=True)
    parser.add_argument("--penalty_csv", type=Path, required=True)
    parser.add_argument("--baseline_model_dir", type=Path, required=True)
    parser.add_argument("--penalty_model_dir", type=Path, required=True)
    parser.add_argument("--base_seed", type=int, default=1)
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--episode_length", type=int, default=25)
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("behavior_evaluation/representative_trajectory_selection"),
    )
    args = parser.parse_args(argv)
    if args.top_k <= 0:
        parser.error("--top_k must be positive.")
    if args.episode_length != 25:
        parser.error("Representative selection requires --episode_length 25.")
    return args


def read_metrics(path: Path) -> List[Dict[str, str]]:
    """Read one nonempty, ordered episode-metrics CSV."""
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Episode metrics CSV does not exist: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = REQUIRED_FIELDS - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path} is missing fields: {sorted(missing)}")
        rows = list(reader)
    if not rows:
        raise ValueError(f"Episode metrics CSV is empty: {path}")
    episode_ids = [int(row["episode"]) for row in rows]
    if episode_ids != list(range(1, len(rows) + 1)):
        raise ValueError(f"Episodes must be ordered and contiguous in: {path}")
    return rows


def vector(rows: Sequence[Mapping[str, str]], field: str) -> np.ndarray:
    """Convert one metric column to finite floating-point values."""
    result = np.asarray([float(row[field]) for row in rows], dtype=float)
    if not np.all(np.isfinite(result)):
        raise ValueError(f"Non-finite values in metric: {field}")
    return result


def absolute_z(values: np.ndarray) -> np.ndarray:
    """Return absolute deviations from the mean in sample-SD units."""
    scale = float(np.std(values, ddof=1))
    if np.isclose(scale, 0.0):
        return np.zeros_like(values)
    return np.abs((values - np.mean(values)) / scale)


def rank_candidates(
    baseline: Sequence[Mapping[str, str]],
    penalty: Sequence[Mapping[str, str]],
    base_seed: int,
) -> List[Dict[str, float | int]]:
    """Rank paired seeds by duplicate reduction and overall typicality.

    A 100-point priority term places every episode with lower Penalty duplicate
    occupancy ahead of episodes without a reduction. Within that group, the
    score primarily rewards reward/distance-coverage centrality and gives only
    a small tie-breaking weight to the magnitude of duplicate improvement.
    """
    if len(baseline) != len(penalty):
        raise ValueError("Baseline and Penalty CSVs have different episode counts.")
    baseline_reward = vector(baseline, "episode_reward")
    penalty_reward = vector(penalty, "episode_reward")
    baseline_duplicate = vector(baseline, "duplicate_rate")
    penalty_duplicate = vector(penalty, "duplicate_rate")
    baseline_distance = vector(baseline, "avg_distance_coverage")
    penalty_distance = vector(penalty, "avg_distance_coverage")

    centrality_penalty = (
        absolute_z(baseline_reward)
        + absolute_z(penalty_reward)
        + absolute_z(baseline_distance)
        + absolute_z(penalty_distance)
    ) / 4.0
    improvement = baseline_duplicate - penalty_duplicate
    positive_scale = float(np.max(np.maximum(improvement, 0.0)))
    normalized_improvement = (
        np.maximum(improvement, 0.0) / positive_scale
        if positive_scale > 0 else np.zeros_like(improvement)
    )
    priority = (improvement > 0).astype(float)
    scores = 100.0 * priority - centrality_penalty + 0.25 * normalized_improvement

    candidates: List[Dict[str, float | int]] = []
    for index in range(len(baseline)):
        candidates.append({
            "seed": base_seed + index,
            "baseline_reward": float(baseline_reward[index]),
            "penalty_reward": float(penalty_reward[index]),
            "baseline_duplicate": float(baseline_duplicate[index]),
            "penalty_duplicate": float(penalty_duplicate[index]),
            "baseline_distance_coverage": float(baseline_distance[index]),
            "penalty_distance_coverage": float(penalty_distance[index]),
            "score": float(scores[index]),
        })
    return sorted(candidates, key=lambda row: (-float(row["score"]), int(row["seed"])))


def write_candidates(path: Path, rows: Sequence[Mapping[str, float | int]]) -> None:
    """Write all paired episodes in descending score order."""
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def run_trajectory(
    script: Path,
    model_dir: Path,
    model_name: str,
    seed: int,
    episode_length: int,
    output_dir: Path,
) -> None:
    """Invoke the existing shared-state trajectory visualization process."""
    command = [
        sys.executable,
        str(script),
        "--env_name", "MPE",
        "--scenario_name", "simple_spread",
        "--algorithm_name", "mappo",
        "--num_agents", "3",
        "--num_landmarks", "3",
        "--model_dir", str(model_dir.expanduser().resolve()),
        "--model_name", model_name,
        "--seed", str(seed),
        "--episode_length", str(episode_length),
        "--output_dir", str(output_dir),
    ]
    subprocess.run(command, check=True)


def render_top_seeds(args: argparse.Namespace, rows: Sequence[Mapping[str, float | int]]) -> None:
    """Render both actors from a shared initial-state snapshot for each top seed."""
    trajectory_script = Path(__file__).resolve().with_name("plot_agent_trajectory.py")
    if not trajectory_script.is_file():
        raise FileNotFoundError(
            f"Trajectory visualization script was not found: {trajectory_script}"
        )
    for row in rows[:args.top_k]:
        seed = int(row["seed"])
        seed_dir = args.output_dir / f"seed_{seed:02d}"
        run_trajectory(
            trajectory_script, args.baseline_model_dir, "Baseline",
            seed, args.episode_length, seed_dir,
        )
        run_trajectory(
            trajectory_script, args.penalty_model_dir, "Penalty 0.1",
            seed, args.episode_length, seed_dir,
        )
        source = seed_dir / "trajectory_comparison.png"
        destination = args.output_dir / f"seed_{seed:02d}_comparison.png"
        if not source.is_file():
            raise RuntimeError(f"Expected comparison figure was not created: {source}")
        shutil.copy2(source, destination)


def main(argv: Sequence[str] | None = None) -> None:
    """Rank all paired seeds, print the top five, and render their trajectories."""
    args = parse_args(argv)
    args.output_dir = args.output_dir.expanduser().resolve()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(
            f"Refusing to overwrite existing selection results: {args.output_dir}"
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    baseline = read_metrics(args.baseline_csv)
    penalty = read_metrics(args.penalty_csv)
    ranked = rank_candidates(baseline, penalty, args.base_seed)
    write_candidates(args.output_dir / "candidate_seeds.csv", ranked)

    print("Top representative seeds:")
    for rank, row in enumerate(ranked[:args.top_k], start=1):
        print(
            f"{rank}. seed={row['seed']} score={float(row['score']):.6f} "
            f"duplicate={float(row['baseline_duplicate']):.6f}"
            f"->{float(row['penalty_duplicate']):.6f} "
            f"distance={float(row['baseline_distance_coverage']):.6f}"
            f"/{float(row['penalty_distance_coverage']):.6f}"
        )
    render_top_seeds(args, ranked)
    print(f"Results: {args.output_dir}")


if __name__ == "__main__":
    main()
