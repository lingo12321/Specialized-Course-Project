#!/usr/bin/env python
"""Evaluate spatial behavior of a trained MAPPO policy in MPE Simple Spread."""

from __future__ import annotations

import csv
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch

from onpolicy.algorithms.r_mappo.algorithm.rMAPPOPolicy import R_MAPPOPolicy
from onpolicy.config import get_config
from onpolicy.envs.mpe.MPE_env import MPEEnv


EPISODE_FIELDS = [
    "model_name", "episode", "episode_reward", "avg_coverage",
    "final_coverage", "success", "full_coverage_step", "duplicate_rate",
    "duplicate_step_rate", "duplicate_event_count", "collision_count",
    "collision_step_rate",
]
TIMESTEP_FIELDS = [
    "model_name", "episode", "timestep", "coverage", "duplicate_rate",
    "collision_pair_count",
]
SUMMARY_METRICS = [
    "episode_reward", "avg_coverage", "final_coverage", "success",
    "full_coverage_step", "duplicate_rate", "duplicate_step_rate",
    "duplicate_event_count", "collision_count", "collision_step_rate",
]
# Legacy constants above remain available for evaluate_random_policy.py.
BEHAVIOR_EPISODE_FIELDS = [
    "model_name", "episode", "episode_reward", "avg_coverage",
    "final_coverage", "hard_success", "avg_soft_coverage",
    "final_soft_coverage", "soft_success", "avg_distance_coverage",
    "final_distance_coverage", "full_coverage_step", "duplicate_rate",
    "duplicate_step_rate", "duplicate_event_count", "collision_count",
    "collision_step_rate",
]
BEHAVIOR_TIMESTEP_FIELDS = [
    "model_name", "episode", "timestep", "coverage", "soft_coverage",
    "distance_coverage", "duplicate_rate", "collision_pair_count",
]
BEHAVIOR_SUMMARY_METRICS = [
    "episode_reward", "avg_coverage", "final_coverage", "hard_success",
    "avg_soft_coverage", "final_soft_coverage", "soft_success",
    "avg_distance_coverage", "final_distance_coverage", "full_coverage_step",
    "duplicate_rate", "duplicate_step_rate", "duplicate_event_count",
    "collision_count", "collision_step_rate",
]
DISTANCE_COVERAGE_D_MAX = 2.0


def parse_args(argv: Sequence[str]) -> Any:
    """Parse evaluation arguments while retaining the project's MAPPO options."""
    parser = get_config()
    parser.add_argument("--scenario_name", default="simple_spread")
    parser.add_argument("--num_agents", type=int, default=3)
    parser.add_argument("--num_landmarks", type=int, default=3)
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--num_eval_episodes", type=int, default=100)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument(
        "--occupancy_mode",
        choices=("landmark_radius", "contact", "fixed"),
        default="landmark_radius",
    )
    parser.add_argument("--occupancy_threshold", type=float, default=None)
    args = parser.parse_args(argv)

    if not args.model_dir:
        parser.error("--model_dir is required.")
    if args.n_rollout_threads != 1:
        parser.error(
            "Behavior evaluation currently supports only single environment thread."
        )
    if args.scenario_name != "simple_spread":
        parser.error("Behavior evaluation requires --scenario_name simple_spread.")
    if args.num_agents != 3 or args.num_landmarks != 3:
        parser.error("Behavior evaluation requires exactly 3 agents and 3 landmarks.")
    if args.algorithm_name != "mappo":
        parser.error("This evaluator currently requires --algorithm_name mappo.")
    if not args.share_policy:
        parser.error("The supplied checkpoints require the shared-policy configuration.")
    if args.num_eval_episodes <= 0 or args.episode_length <= 0:
        parser.error("Episode count and episode length must be positive.")
    if args.occupancy_mode == "fixed":
        if args.occupancy_threshold is None or args.occupancy_threshold <= 0:
            parser.error(
                "--occupancy_mode fixed requires a positive --occupancy_threshold."
            )
    elif args.occupancy_threshold is not None:
        parser.error(
            "--occupancy_threshold is only valid with --occupancy_mode fixed."
        )

    # train_mpe.py enforces these values for MAPPO.
    args.use_recurrent_policy = False
    args.use_naive_recurrent_policy = False
    return args


def prepare_output_dir(requested: Path) -> Path:
    """Create an output directory without overwriting an existing nonempty run."""
    requested = requested.expanduser().resolve()
    if requested.exists() and any(requested.iterdir()):
        stamp = time.strftime("%Y%m%d_%H%M%S")
        requested = requested.with_name(f"{requested.name}_{stamp}")
        print(f"Output directory is not empty; writing to: {requested}")
    requested.mkdir(parents=True, exist_ok=True)
    return requested


def validate_checkpoint(model_dir: Path) -> Path:
    """Return the actor checkpoint path or raise a clear validation error."""
    model_dir = model_dir.expanduser().resolve()
    if not model_dir.is_dir():
        raise FileNotFoundError(f"Model directory does not exist: {model_dir}")
    actor_path = model_dir / "actor.pt"
    if not actor_path.is_file():
        raise FileNotFoundError(
            f"Shared MAPPO actor checkpoint was not found: {actor_path}"
        )
    return actor_path


def occupancy_definition(mode: str, fixed_threshold: float | None) -> str:
    """Describe the selected occupancy rule for metadata."""
    if mode == "landmark_radius":
        return "distance(agent, landmark) < landmark.size"
    if mode == "contact":
        return "distance(agent, landmark) < agent.size + landmark.size"
    return f"distance(agent, landmark) < {fixed_threshold}"


def threshold_for(agent: Any, landmark: Any, args: Any) -> float:
    """Return the occupancy threshold for one agent-landmark pair."""
    if args.occupancy_mode == "landmark_radius":
        return float(landmark.size)
    if args.occupancy_mode == "contact":
        return float(agent.size + landmark.size)
    return float(args.occupancy_threshold)


def spatial_metrics(world: Any, args: Any) -> Tuple[float, float, int]:
    """Compute coverage, duplicate rate, and unique colliding agent pairs."""
    agents = world.agents
    landmarks = world.landmarks
    if len(agents) != 3 or len(landmarks) != 3:
        raise RuntimeError(
            "Environment world does not contain the required 3 agents and 3 landmarks."
        )

    occupancies: List[int] = []
    for landmark in landmarks:
        nearby = 0
        for agent in agents:
            distance = np.linalg.norm(agent.state.p_pos - landmark.state.p_pos)
            nearby += int(distance < threshold_for(agent, landmark, args))
        occupancies.append(nearby)

    covered = sum(count >= 1 for count in occupancies)
    duplicated = sum(count >= 2 for count in occupancies)
    collision_pairs = 0
    # A continuing collision contributes one pair-step on every timestep.
    for i in range(len(agents)):
        for j in range(i + 1, len(agents)):
            distance = np.linalg.norm(
                agents[i].state.p_pos - agents[j].state.p_pos
            )
            collision_pairs += int(
                distance < agents[i].size + agents[j].size
            )

    return (
        covered / len(landmarks),
        duplicated / len(landmarks),
        collision_pairs,
    )


def soft_coverage_score(world: Any, args: Any) -> float:
    """Return mean linear distance completion across all landmarks.

    For landmark j, d_j is its distance to the nearest agent and its score is
    max(0, 1 - d_j / D). D is the active occupancy threshold for that nearest
    agent-landmark pair.
    """
    scores: List[float] = []
    for landmark in world.landmarks:
        nearest_agent = min(
            world.agents,
            key=lambda agent: np.linalg.norm(
                agent.state.p_pos - landmark.state.p_pos
            ),
        )
        distance = float(np.linalg.norm(
            nearest_agent.state.p_pos - landmark.state.p_pos
        ))
        distance_scale = threshold_for(nearest_agent, landmark, args)
        if distance_scale <= 0:
            raise RuntimeError("Soft coverage requires a positive distance scale.")
        scores.append(max(0.0, 1.0 - distance / distance_scale))
    return float(np.mean(scores))


def distance_coverage_score(world: Any) -> float:
    """Return landmark proximity normalized by the fixed D_max of 2.0."""
    scores: List[float] = []
    for landmark in world.landmarks:
        nearest_distance = min(
            float(np.linalg.norm(agent.state.p_pos - landmark.state.p_pos))
            for agent in world.agents
        )
        scores.append(
            max(0.0, 1.0 - nearest_distance / DISTANCE_COVERAGE_D_MAX)
        )
    return float(np.mean(scores))


def discrete_actions(policy_actions: np.ndarray, env: Any) -> List[np.ndarray]:
    """Convert policy action indices to the one-hot actions expected by MPEEnv."""
    action_list: List[np.ndarray] = []
    for agent_id, raw_action in enumerate(policy_actions):
        space = env.action_space[agent_id]
        values = np.asarray(raw_action, dtype=np.int64).reshape(-1)
        if space.__class__.__name__ == "Discrete":
            if values.size != 1:
                raise RuntimeError(
                    f"Unexpected Discrete action shape for agent {agent_id}: "
                    f"{values.shape}"
                )
            action_list.append(np.eye(space.n, dtype=np.float32)[values[0]])
        elif space.__class__.__name__ == "MultiDiscrete":
            pieces = [
                np.eye(space.high[i] + 1, dtype=np.float32)[values[i]]
                for i in range(space.shape)
            ]
            action_list.append(np.concatenate(pieces))
        else:
            raise NotImplementedError(
                f"Unsupported MPE action space: {space.__class__.__name__}"
            )
    return action_list


def load_policy(args: Any, env: Any, actor_path: Path, device: torch.device) -> Any:
    """Construct the existing MAPPO policy and restore its actor checkpoint."""
    share_obs_space = (
        env.share_observation_space[0]
        if args.use_centralized_V
        else env.observation_space[0]
    )
    policy = R_MAPPOPolicy(
        args,
        env.observation_space[0],
        share_obs_space,
        env.action_space[0],
        device=device,
    )
    try:
        state_dict = torch.load(str(actor_path), map_location=device)
        policy.actor.load_state_dict(state_dict)
    except RuntimeError as exc:
        raise RuntimeError(
            "Actor checkpoint is incompatible with the requested environment or "
            "MAPPO configuration (expected 3 agents, 3 landmarks, simple_spread). "
            f"Original error: {exc}"
        ) from exc
    policy.actor.eval()
    return policy


def summarize(
    rows: Sequence[Dict[str, Any]],
    metrics: Sequence[str] = SUMMARY_METRICS,
) -> Dict[str, Dict[str, float]]:
    """Compute per-metric mean and sample standard deviation."""
    result: Dict[str, Dict[str, float]] = {}
    for metric in metrics:
        values = np.asarray([float(row[metric]) for row in rows], dtype=np.float64)
        result[metric] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
        }
    return result


def write_csv(path: Path, fields: Sequence[str], rows: Sequence[Dict[str, Any]]) -> None:
    """Write dictionaries to a UTF-8 CSV with a stable column order."""
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def run(args: Any) -> Path:
    """Run all evaluation episodes and write machine-readable results."""
    model_dir = Path(args.model_dir)
    actor_path = validate_checkpoint(model_dir)
    output_dir = prepare_output_dir(args.output_dir)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda:0" if args.cuda and torch.cuda.is_available() else "cpu")
    env = MPEEnv(args)
    if not hasattr(env, "world"):
        raise RuntimeError("MPE environment does not expose the required world state.")
    policy = load_policy(args, env, actor_path, device)

    episode_rows: List[Dict[str, Any]] = []
    timestep_rows: List[Dict[str, Any]] = []
    started = time.time()
    try:
        for episode_index in range(args.num_eval_episodes):
            episode_seed = args.seed + episode_index
            env.seed(episode_seed)
            obs = np.asarray(env.reset(), dtype=np.float32)
            rnn_states = np.zeros(
                (args.num_agents, args.recurrent_N, args.hidden_size),
                dtype=np.float32,
            )
            masks = np.ones((args.num_agents, 1), dtype=np.float32)
            rewards: List[float] = []
            coverages: List[float] = []
            soft_coverages: List[float] = []
            distance_coverages: List[float] = []
            duplicate_rates: List[float] = []
            collision_counts: List[int] = []
            full_coverage_step = -1

            for timestep in range(1, args.episode_length + 1):
                with torch.no_grad():
                    action, next_rnn_states = policy.act(
                        obs, rnn_states, masks, deterministic=True
                    )
                action_array = action.detach().cpu().numpy()
                rnn_states = next_rnn_states.detach().cpu().numpy()
                obs_next, reward_n, done_n, _ = env.step(
                    discrete_actions(action_array, env)
                )

                # Direct MPEEnv does not auto-reset, so world still represents
                # the current episode's post-action state here.
                coverage, duplicate_rate, collision_pairs = spatial_metrics(
                    env.world, args
                )
                soft_coverage = soft_coverage_score(env.world, args)
                distance_coverage = distance_coverage_score(env.world)
                team_reward = float(np.asarray(reward_n, dtype=float).reshape(-1)[0])
                rewards.append(team_reward)
                coverages.append(coverage)
                soft_coverages.append(soft_coverage)
                distance_coverages.append(distance_coverage)
                duplicate_rates.append(duplicate_rate)
                collision_counts.append(collision_pairs)
                if full_coverage_step < 0 and np.isclose(coverage, 1.0):
                    full_coverage_step = timestep

                timestep_rows.append({
                    "model_name": args.model_name,
                    "episode": episode_index + 1,
                    "timestep": timestep,
                    "coverage": coverage,
                    "soft_coverage": soft_coverage,
                    "distance_coverage": distance_coverage,
                    "duplicate_rate": duplicate_rate,
                    "collision_pair_count": collision_pairs,
                })
                obs = np.asarray(obs_next, dtype=np.float32)
                done = np.asarray(done_n, dtype=bool)
                if np.any(done) and not np.all(done):
                    raise RuntimeError("Simple Spread agents ended asynchronously.")
                if np.all(done) and timestep != args.episode_length:
                    raise RuntimeError(
                        "Environment ended before the requested episode length."
                    )
                masks[:] = 0.0 if np.all(done) else 1.0

            final_coverage = coverages[-1]
            episode_rows.append({
                "model_name": args.model_name,
                "episode": episode_index + 1,
                "episode_reward": float(np.sum(rewards)),
                "avg_coverage": float(np.mean(coverages)),
                "final_coverage": final_coverage,
                "hard_success": int(np.isclose(final_coverage, 1.0)),
                "avg_soft_coverage": float(np.mean(soft_coverages)),
                "final_soft_coverage": soft_coverages[-1],
                "soft_success": soft_coverages[-1],
                "avg_distance_coverage": float(np.mean(distance_coverages)),
                "final_distance_coverage": distance_coverages[-1],
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

    summary = summarize(episode_rows, BEHAVIOR_SUMMARY_METRICS)
    write_csv(
        output_dir / "episode_metrics.csv",
        BEHAVIOR_EPISODE_FIELDS,
        episode_rows,
    )
    write_csv(
        output_dir / "timestep_metrics.csv",
        BEHAVIOR_TIMESTEP_FIELDS,
        timestep_rows,
    )
    summary_names = {
        "hard_success": "hard_success_rate",
        "soft_success": "soft_success_mean",
    }
    summary_rows = [
        {
            "metric": summary_names.get(metric, metric),
            "mean": stats["mean"],
            "std": stats["std"],
        }
        for metric, stats in summary.items()
    ]
    write_csv(output_dir / "summary.csv", ("metric", "mean", "std"), summary_rows)

    metadata = {
        "model_name": args.model_name,
        "model_dir": str(model_dir.expanduser().resolve()),
        "actor_checkpoint": str(actor_path),
        "parameters": {
            "num_eval_episodes": args.num_eval_episodes,
            "seed": args.seed,
            "episode_seed_rule": "seed + zero_based_episode_index",
            "episode_length": args.episode_length,
            "deterministic": True,
            "n_rollout_threads": args.n_rollout_threads,
        },
        "environment": {
            "env_name": args.env_name,
            "scenario_name": args.scenario_name,
            "algorithm_name": args.algorithm_name,
            "num_agents": args.num_agents,
            "num_landmarks": args.num_landmarks,
            "share_policy": args.share_policy,
            "use_centralized_V": args.use_centralized_V,
        },
        "occupancy": {
            "mode": args.occupancy_mode,
            "fixed_threshold": args.occupancy_threshold,
            "definition": occupancy_definition(
                args.occupancy_mode, args.occupancy_threshold
            ),
            "comparison": "strict (<), not inclusive (<=)",
        },
        "distance_coverage": {
            "d_max": DISTANCE_COVERAGE_D_MAX,
            "definition": (
                "mean_j max(0, 1 - min_i distance(agent_i, landmark_j) / 2.0)"
            ),
        },
        "reward_definition": (
            "Sum over timesteps of one representative per-agent return; MPE "
            "collaborative mode returns the same summed team reward to every agent."
        ),
        "collision_definition": (
            "Unique unordered agent pairs per timestep; a five-step continuing "
            "collision contributes five pair-step collisions."
        ),
        "metrics": summary,
        "runtime_seconds": time.time() - started,
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, ensure_ascii=False, allow_nan=False)

    print(f"\nModel: {args.model_name}")
    print(f"Episodes: {args.num_eval_episodes}")
    for metric in (
        "episode_reward", "avg_coverage", "final_coverage",
        "avg_soft_coverage", "final_soft_coverage", "hard_success",
        "avg_distance_coverage", "final_distance_coverage",
        "duplicate_step_rate", "collision_count",
    ):
        stats = summary[metric]
        label = "hard_success_rate" if metric == "hard_success" else metric
        print(f"{label}: {stats['mean']:.6f} ± {stats['std']:.6f}")
    print(f"Results: {output_dir}")
    return output_dir


def main(argv: Sequence[str] | None = None) -> None:
    """CLI entry point."""
    args = parse_args(sys.argv[1:] if argv is None else argv)
    run(args)


if __name__ == "__main__":
    main()
