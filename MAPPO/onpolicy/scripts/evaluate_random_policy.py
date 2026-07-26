#!/usr/bin/env python
"""Evaluate a random MPE policy and compare it with paired MAPPO results."""

from __future__ import annotations

import argparse
import csv
import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

import numpy as np

from onpolicy.config import get_config
from onpolicy.envs.mpe.MPE_env import MPEEnv
from onpolicy.scripts.evaluate_agent_behavior import (
    EPISODE_FIELDS,
    SUMMARY_METRICS,
    discrete_actions,
    spatial_metrics,
    summarize,
    write_csv,
)


REPORT_METRICS = (
    ("episode_reward", "Episode reward", "higher"),
    ("avg_coverage", "Average coverage", "higher"),
    ("final_coverage", "Final coverage", "higher"),
    ("success", "Success rate", "higher"),
    ("duplicate_step_rate", "Duplicate step rate", "lower"),
    ("collision_count", "Collision count", "lower"),
)


def parse_args(argv: Sequence[str]) -> Any:
    """Parse project-compatible environment and random-evaluation options."""
    parser = get_config()
    parser.add_argument("--scenario_name", default="simple_spread")
    parser.add_argument("--num_agents", type=int, default=3)
    parser.add_argument("--num_landmarks", type=int, default=3)
    parser.add_argument("--num_eval_episodes", type=int, default=100)
    parser.add_argument("--output_dir", type=Path, default=Path("random_policy_evaluation"))
    parser.add_argument(
        "--mappo_csv",
        type=Path,
        required=True,
        help="Paired MAPPO episode_metrics.csv evaluated with fixed threshold 0.2.",
    )
    args = parser.parse_args(argv)
    if args.scenario_name != "simple_spread":
        parser.error("Random policy sanity check requires simple_spread.")
    if args.num_agents != 3 or args.num_landmarks != 3:
        parser.error("Random policy sanity check requires 3 agents and 3 landmarks.")
    if args.n_rollout_threads != 1:
        parser.error(
            "Behavior evaluation currently supports only single environment thread."
        )
    if args.num_eval_episodes != 100:
        parser.error("Policy sanity check requires exactly 100 episodes.")
    if args.episode_length != 25:
        parser.error("Policy sanity check requires --episode_length 25.")
    args.occupancy_mode = "fixed"
    args.occupancy_threshold = 0.2
    return args


def prepare_output_dir(path: Path) -> Path:
    """Create a result directory without overwriting a previous evaluation."""
    path = path.expanduser().resolve()
    if path.exists() and any(path.iterdir()):
        path = path.with_name(f"{path.name}_{time.strftime('%Y%m%d_%H%M%S')}")
        print(f"Output directory is not empty; writing to: {path}")
    path.mkdir(parents=True, exist_ok=True)
    return path


def sample_actions(env: Any) -> List[np.ndarray]:
    """Sample one legal action per agent from the current action spaces."""
    sampled = [
        np.asarray(space.sample(), dtype=np.int64).reshape(-1)
        for space in env.action_space
    ]
    return discrete_actions(np.asarray(sampled), env)


def evaluate_random(args: Any) -> List[Dict[str, Any]]:
    """Run the random policy with the same episode seed schedule as MAPPO."""
    env = MPEEnv(args)
    rows: List[Dict[str, Any]] = []
    try:
        for episode_index in range(args.num_eval_episodes):
            episode_seed = args.seed + episode_index
            env.seed(episode_seed)
            for agent_id, space in enumerate(env.action_space):
                if hasattr(space, "seed"):
                    space.seed(episode_seed * 1000 + agent_id)
            env.reset()

            rewards: List[float] = []
            coverages: List[float] = []
            duplicate_rates: List[float] = []
            collision_counts: List[int] = []
            full_coverage_step = -1

            for timestep in range(1, args.episode_length + 1):
                _, reward_n, done_n, _ = env.step(sample_actions(env))
                coverage, duplicate_rate, collision_pairs = spatial_metrics(
                    env.world, args
                )
                rewards.append(
                    float(np.asarray(reward_n, dtype=float).reshape(-1)[0])
                )
                coverages.append(coverage)
                duplicate_rates.append(duplicate_rate)
                collision_counts.append(collision_pairs)
                if full_coverage_step < 0 and np.isclose(coverage, 1.0):
                    full_coverage_step = timestep
                done = np.asarray(done_n, dtype=bool)
                if np.any(done) and not np.all(done):
                    raise RuntimeError("Simple Spread agents ended asynchronously.")
                if np.all(done) and timestep != args.episode_length:
                    raise RuntimeError("Environment ended before episode_length.")

            final_coverage = coverages[-1]
            rows.append({
                "model_name": "Random",
                "episode": episode_index + 1,
                "episode_reward": float(np.sum(rewards)),
                "avg_coverage": float(np.mean(coverages)),
                "final_coverage": final_coverage,
                "success": int(np.isclose(final_coverage, 1.0)),
                "full_coverage_step": full_coverage_step,
                "duplicate_rate": float(np.mean(duplicate_rates)),
                "duplicate_step_rate": float(
                    np.mean(np.asarray(duplicate_rates) > 0)
                ),
                "duplicate_event_count": int(
                    np.sum(np.asarray(duplicate_rates) * args.num_landmarks)
                ),
                "collision_count": int(np.sum(collision_counts)),
                "collision_step_rate": float(
                    np.mean(np.asarray(collision_counts) > 0)
                ),
            })
    finally:
        env.close()
    return rows


def read_mappo_rows(path: Path, expected_episodes: int) -> List[Dict[str, str]]:
    """Read and validate paired MAPPO episode metrics."""
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"MAPPO episode CSV does not exist: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = set(reader.fieldnames or [])
        required = set(EPISODE_FIELDS)
        if missing := required - fields:
            raise ValueError(f"MAPPO CSV is missing fields: {sorted(missing)}")
        rows = list(reader)
    if len(rows) != expected_episodes:
        raise ValueError(
            f"MAPPO CSV contains {len(rows)} episodes; expected {expected_episodes}."
        )
    episode_ids = [int(row["episode"]) for row in rows]
    if episode_ids != list(range(1, expected_episodes + 1)):
        raise ValueError("MAPPO CSV episode IDs are not the required ordered 1–100.")
    return rows


def values(rows: Sequence[Mapping[str, Any]], metric: str) -> np.ndarray:
    """Extract a finite numeric metric vector."""
    result = np.asarray([float(row[metric]) for row in rows], dtype=float)
    if not np.all(np.isfinite(result)):
        raise ValueError(f"Metric contains non-finite values: {metric}")
    return result


def paired_result(
    mappo_rows: Sequence[Mapping[str, Any]],
    random_rows: Sequence[Mapping[str, Any]],
    metric: str,
) -> Dict[str, float | bool]:
    """Return paired mean difference and an approximate 95% confidence interval."""
    difference = values(mappo_rows, metric) - values(random_rows, metric)
    mean = float(np.mean(difference))
    std = float(np.std(difference, ddof=1))
    standard_error = std / math.sqrt(len(difference))
    # For 99 degrees of freedom, the two-sided 95% t critical value is 1.984.
    half_width = 1.984 * standard_error
    return {
        "mean_difference": mean,
        "ci_low": mean - half_width,
        "ci_high": mean + half_width,
        "ci_excludes_zero": bool(mean - half_width > 0 or mean + half_width < 0),
    }


def write_report(
    path: Path,
    mappo_rows: Sequence[Mapping[str, Any]],
    random_rows: Sequence[Mapping[str, Any]],
) -> None:
    """Write a cautious MAPPO-versus-random sanity-check report."""
    lines = [
        "# MAPPO Policy Sanity Check",
        "",
        "MAPPO and Random were evaluated on paired Simple Spread initial states "
        "(seeds 1–100), with 25 steps and occupancy defined as distance < 0.2.",
        "",
        "| Metric | MAPPO mean | Random mean | MAPPO − Random | Approx. 95% paired CI |",
        "|---|---:|---:|---:|---:|",
    ]
    results: Dict[str, Dict[str, float | bool]] = {}
    for metric, label, _ in REPORT_METRICS:
        result = paired_result(mappo_rows, random_rows, metric)
        results[metric] = result
        lines.append(
            f"| {label} | {np.mean(values(mappo_rows, metric)):.6f} | "
            f"{np.mean(values(random_rows, metric)):.6f} | "
            f"{result['mean_difference']:.6f} | "
            f"[{result['ci_low']:.6f}, {result['ci_high']:.6f}] |"
        )

    reward_better = float(results["episode_reward"]["ci_low"]) > 0
    coverage_better = float(results["avg_coverage"]["ci_low"]) > 0
    collision_lower = float(results["collision_count"]["ci_high"]) < 0
    lines.extend(["", "## Interpretation", ""])
    lines.append(
        "- MAPPO reward is clearly higher than Random under this paired check."
        if reward_better else
        "- MAPPO reward is not clearly higher than Random under this paired check."
    )
    lines.append(
        "- MAPPO average coverage is clearly higher than Random."
        if coverage_better else
        "- MAPPO average coverage is not clearly higher than Random."
    )
    lines.append(
        "- MAPPO collision count is clearly lower than Random."
        if collision_lower else
        "- MAPPO collision count is not clearly lower than Random."
    )
    if reward_better and coverage_better:
        lines.append(
            "- The reward and coverage results provide evidence that MAPPO learned "
            "behavior beyond random exploration. They do not, by themselves, prove "
            "optimal or robust cooperation."
        )
    else:
        lines.append(
            "- The evaluated policy does not show a consistently clear advantage "
            "over Random on both reward and coverage, so this check alone does not "
            "support a strong claim of effective cooperative behavior."
        )
    lines.extend([
        "",
        "The intervals are descriptive paired t intervals across 100 seeds and are "
        "included to avoid treating small raw mean differences as automatically "
        "meaningful. This is a policy sanity check, not a causal experiment.",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> None:
    """Run random evaluation and write summary and paired policy comparison."""
    args = parse_args(sys.argv[1:] if argv is None else argv)
    output_dir = prepare_output_dir(args.output_dir)
    mappo_rows = read_mappo_rows(args.mappo_csv, args.num_eval_episodes)
    random_rows = evaluate_random(args)
    write_csv(output_dir / "episode_metrics.csv", EPISODE_FIELDS, random_rows)
    summary = summarize(random_rows)
    write_csv(
        output_dir / "summary.csv",
        ("metric", "mean", "std"),
        [
            {"metric": metric, "mean": summary[metric]["mean"],
             "std": summary[metric]["std"]}
            for metric in SUMMARY_METRICS
        ],
    )
    write_report(
        output_dir / "policy_comparison.md", mappo_rows, random_rows
    )
    print("Random policy evaluation complete.")
    print(f"Results: {output_dir}")


if __name__ == "__main__":
    main()
