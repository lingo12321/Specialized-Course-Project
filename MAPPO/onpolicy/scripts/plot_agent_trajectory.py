#!/usr/bin/env python
"""Plot one deterministic MAPPO trajectory in MPE Simple Spread."""

from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from onpolicy.config import get_config
from onpolicy.envs.mpe.MPE_env import MPEEnv
from onpolicy.scripts.evaluate_agent_behavior import (
    discrete_actions,
    load_policy,
    validate_checkpoint,
)


AGENT_COLORS = ("tab:blue", "tab:orange", "tab:green")
TRAJECTORY_FIELDS = [
    "model_name", "seed", "timestep",
    "agent0_x", "agent0_y",
    "agent1_x", "agent1_y",
    "agent2_x", "agent2_y",
]
LANDMARK_FIELDS = [
    "model_name", "seed", "landmark",
    "landmark_x", "landmark_y",
]


def parse_args(argv: Sequence[str]) -> Any:
    """Parse trajectory options using the existing MAPPO configuration."""
    parser = get_config()
    parser.add_argument("--scenario_name", default="simple_spread")
    parser.add_argument("--num_agents", type=int, default=3)
    parser.add_argument("--num_landmarks", type=int, default=3)
    parser.add_argument("--model_name", required=True)
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("behavior_evaluation/trajectory_visualization"),
    )
    args = parser.parse_args(argv)
    if not args.model_dir:
        parser.error("--model_dir is required.")
    if args.scenario_name != "simple_spread":
        parser.error("Trajectory visualization requires simple_spread.")
    if args.num_agents != 3 or args.num_landmarks != 3:
        parser.error("Trajectory visualization requires 3 agents and 3 landmarks.")
    if args.algorithm_name != "mappo" or not args.share_policy:
        parser.error("Trajectory visualization requires shared-policy MAPPO.")
    if args.episode_length <= 0:
        parser.error("--episode_length must be positive.")
    args.use_recurrent_policy = False
    args.use_naive_recurrent_policy = False
    return args


def model_slug(model_name: str) -> str:
    """Return a stable output name, recognizing the two requested model labels."""
    lowered = model_name.strip().lower()
    if "baseline" in lowered:
        return "baseline"
    if "penalty" in lowered:
        return "penalty"
    slug = re.sub(r"[^a-z0-9]+", "_", lowered).strip("_")
    if not slug:
        raise ValueError("model_name must contain at least one letter or digit.")
    return slug


def positions(world: Any) -> np.ndarray:
    """Copy the three current agent positions as a (3, 2) array."""
    if len(world.agents) != 3 or len(world.landmarks) != 3:
        raise RuntimeError("World must contain exactly 3 agents and 3 landmarks.")
    return np.asarray(
        [np.asarray(agent.state.p_pos, dtype=float).copy()
         for agent in world.agents]
    )


def capture_world_state(world: Any) -> Dict[str, np.ndarray]:
    """Copy all dynamic entity state needed to reproduce the reset state."""
    return {
        "agent_p_pos": np.asarray(
            [agent.state.p_pos.copy() for agent in world.agents]
        ),
        "agent_p_vel": np.asarray(
            [agent.state.p_vel.copy() for agent in world.agents]
        ),
        "agent_c": np.asarray(
            [agent.state.c.copy() for agent in world.agents]
        ),
        "landmark_p_pos": np.asarray(
            [landmark.state.p_pos.copy() for landmark in world.landmarks]
        ),
        "landmark_p_vel": np.asarray(
            [landmark.state.p_vel.copy() for landmark in world.landmarks]
        ),
        "world_step": np.asarray(world.world_step, dtype=np.int64),
    }


def restore_world_state(env: Any, state: Mapping[str, np.ndarray]) -> None:
    """Restore a saved reset state into an independently created MPE env."""
    if state["agent_p_pos"].shape != (3, 2):
        raise ValueError("Saved initial state does not contain three 2-D agents.")
    if state["landmark_p_pos"].shape != (3, 2):
        raise ValueError("Saved initial state does not contain three 2-D landmarks.")
    for agent_id, agent in enumerate(env.world.agents):
        agent.state.p_pos = state["agent_p_pos"][agent_id].copy()
        agent.state.p_vel = state["agent_p_vel"][agent_id].copy()
        agent.state.c = state["agent_c"][agent_id].copy()
    for landmark_id, landmark in enumerate(env.world.landmarks):
        landmark.state.p_pos = state["landmark_p_pos"][landmark_id].copy()
        landmark.state.p_vel = state["landmark_p_vel"][landmark_id].copy()
    env.world.world_step = int(state["world_step"])
    env.current_step = 0


def shared_initial_state(env: Any, output_dir: Path, seed: int) -> Dict[str, np.ndarray]:
    """Save state_0 once, or restore the exact saved state for another actor."""
    snapshot_path = output_dir / "initial_state.npz"
    if snapshot_path.is_file():
        with np.load(snapshot_path, allow_pickle=False) as snapshot:
            saved_seed = int(snapshot["seed"])
            if saved_seed != seed:
                raise ValueError(
                    f"Saved initial state uses seed {saved_seed}, not requested seed {seed}."
                )
            state = {
                key: snapshot[key].copy()
                for key in (
                    "agent_p_pos", "agent_p_vel", "agent_c",
                    "landmark_p_pos", "landmark_p_vel", "world_step",
                )
            }
        restore_world_state(env, state)
        return state

    state = capture_world_state(env.world)
    np.savez(
        snapshot_path,
        seed=np.asarray(seed, dtype=np.int64),
        **state,
    )
    return state


def rollout(args: Any, output_dir: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Run one actor from the shared saved state_0 and return its trajectory."""
    actor_path = validate_checkpoint(Path(args.model_dir))
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(
        "cuda:0" if args.cuda and torch.cuda.is_available() else "cpu"
    )
    env = MPEEnv(args)
    try:
        env.seed(args.seed)
        obs = np.asarray(env.reset(), dtype=np.float32)
        initial_state = shared_initial_state(env, output_dir, args.seed)
        # Observation depends only on the restored world state. Recompute it
        # after restoration so both actors receive exactly the same first input.
        obs = np.asarray(
            [env._get_obs(agent) for agent in env.agents],
            dtype=np.float32,
        )
        landmarks = np.asarray(
            initial_state["landmark_p_pos"], dtype=float
        )
        policy = load_policy(args, env, actor_path, device)
        rnn_states = np.zeros(
            (args.num_agents, args.recurrent_N, args.hidden_size),
            dtype=np.float32,
        )
        masks = np.ones((args.num_agents, 1), dtype=np.float32)
        trajectory = [positions(env.world)]

        for timestep in range(1, args.episode_length + 1):
            with torch.no_grad():
                action, next_rnn_states = policy.act(
                    obs, rnn_states, masks, deterministic=True
                )
            obs_next, _, done_n, _ = env.step(
                discrete_actions(action.detach().cpu().numpy(), env)
            )
            trajectory.append(positions(env.world))
            obs = np.asarray(obs_next, dtype=np.float32)
            rnn_states = next_rnn_states.detach().cpu().numpy()
            done = np.asarray(done_n, dtype=bool)
            if np.any(done) and not np.all(done):
                raise RuntimeError("Simple Spread agents ended asynchronously.")
            if np.all(done) and timestep != args.episode_length:
                raise RuntimeError("Environment ended before episode_length.")
            masks[:] = 0.0 if np.all(done) else 1.0
        return np.asarray(trajectory), landmarks
    finally:
        env.close()


def write_trajectory(
    path: Path, model_name: str, seed: int, trajectory: np.ndarray
) -> None:
    """Write reset state (t=0) and all post-action states to trajectory.csv."""
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=TRAJECTORY_FIELDS)
        writer.writeheader()
        for timestep, state in enumerate(trajectory):
            row: Dict[str, Any] = {
                "model_name": model_name,
                "seed": seed,
                "timestep": timestep,
            }
            for agent_id in range(3):
                row[f"agent{agent_id}_x"] = float(state[agent_id, 0])
                row[f"agent{agent_id}_y"] = float(state[agent_id, 1])
            writer.writerow(row)


def write_landmarks(
    path: Path, model_name: str, seed: int, landmarks: np.ndarray
) -> None:
    """Write the fixed landmark positions for this episode."""
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=LANDMARK_FIELDS)
        writer.writeheader()
        for landmark_id, point in enumerate(landmarks):
            writer.writerow({
                "model_name": model_name,
                "seed": seed,
                "landmark": landmark_id,
                "landmark_x": float(point[0]),
                "landmark_y": float(point[1]),
            })


def axis_limits(
    trajectories: Sequence[np.ndarray], landmarks: Sequence[np.ndarray]
) -> Tuple[float, float, float, float]:
    """Return shared square plot limits covering every trajectory and landmark."""
    points = np.concatenate(
        [item.reshape(-1, 2) for item in (*trajectories, *landmarks)], axis=0
    )
    x_min, y_min = np.min(points, axis=0)
    x_max, y_max = np.max(points, axis=0)
    center_x = (x_min + x_max) / 2
    center_y = (y_min + y_max) / 2
    half_span = max(x_max - x_min, y_max - y_min) / 2 + 0.15
    return (
        center_x - half_span,
        center_x + half_span,
        center_y - half_span,
        center_y + half_span,
    )


def draw_trajectory(
    axis: Any,
    trajectory: np.ndarray,
    landmarks: np.ndarray,
    title: str,
    limits: Tuple[float, float, float, float],
) -> None:
    """Draw three labeled agent paths and three labeled landmarks."""
    for landmark_id, point in enumerate(landmarks):
        axis.scatter(
            point[0], point[1], marker="*", s=190, c="black",
            edgecolors="gold", linewidths=0.9, zorder=5,
        )
        axis.annotate(
            f"Landmark{landmark_id}", point, xytext=(5, 6),
            textcoords="offset points", fontsize=8,
        )
    for agent_id, color in enumerate(AGENT_COLORS):
        path = trajectory[:, agent_id, :]
        axis.plot(
            path[:, 0], path[:, 1], color=color, linewidth=1.8,
            alpha=0.9, label=f"Agent{agent_id}",
        )
        axis.scatter(
            path[0, 0], path[0, 1], s=65, facecolors="none",
            edgecolors=color, linewidths=1.8, zorder=6,
        )
        axis.scatter(
            path[-1, 0], path[-1, 1], s=55, color=color, zorder=6,
        )
        axis.annotate(
            f"Agent{agent_id}", path[-1], xytext=(5, -11),
            textcoords="offset points", color=color, fontsize=8,
        )
    axis.set_title(title)
    axis.set_xlabel("x")
    axis.set_ylabel("y")
    axis.set_aspect("equal", adjustable="box")
    axis.set_xlim(limits[0], limits[1])
    axis.set_ylim(limits[2], limits[3])
    axis.grid(alpha=0.25)
    axis.legend(loc="best", fontsize=8)


def save_single_plot(
    path: Path,
    trajectory: np.ndarray,
    landmarks: np.ndarray,
    title: str,
) -> None:
    """Save one high-resolution trajectory figure."""
    fig, axis = plt.subplots(figsize=(7, 7))
    draw_trajectory(
        axis, trajectory, landmarks, title,
        axis_limits([trajectory], [landmarks]),
    )
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def read_saved(model_dir: Path) -> Tuple[str, np.ndarray, np.ndarray]:
    """Load one saved model trajectory and landmark CSV pair."""
    with (model_dir / "trajectory.csv").open(
        "r", encoding="utf-8-sig", newline=""
    ) as handle:
        rows = list(csv.DictReader(handle))
    with (model_dir / "landmarks.csv").open(
        "r", encoding="utf-8-sig", newline=""
    ) as handle:
        landmark_rows = list(csv.DictReader(handle))
    if not rows or len(landmark_rows) != 3:
        raise ValueError(f"Incomplete trajectory data in: {model_dir}")
    trajectory = np.asarray([
        [[float(row[f"agent{i}_x"]), float(row[f"agent{i}_y"])]
         for i in range(3)]
        for row in rows
    ])
    landmarks = np.asarray([
        [float(row["landmark_x"]), float(row["landmark_y"])]
        for row in landmark_rows
    ])
    return rows[0]["model_name"], trajectory, landmarks


def maybe_save_comparison(output_dir: Path) -> bool:
    """Create the requested side-by-side figure once both runs are available."""
    baseline_dir = output_dir / "baseline"
    penalty_dir = output_dir / "penalty"
    required = [
        baseline_dir / "trajectory.csv", baseline_dir / "landmarks.csv",
        penalty_dir / "trajectory.csv", penalty_dir / "landmarks.csv",
    ]
    if not all(path.is_file() for path in required):
        return False
    baseline_name, baseline_trajectory, baseline_landmarks = read_saved(
        baseline_dir
    )
    penalty_name, penalty_trajectory, penalty_landmarks = read_saved(penalty_dir)
    baseline_initial = baseline_trajectory[0]
    penalty_initial = penalty_trajectory[0]
    if not np.array_equal(baseline_initial, penalty_initial):
        raise RuntimeError(
            "Baseline and Penalty initial agent positions are not identical."
        )
    if not np.array_equal(baseline_landmarks, penalty_landmarks):
        raise RuntimeError("Baseline and Penalty landmark positions are not identical.")
    print("Baseline initial agent positions:")
    print(baseline_initial)
    print("Penalty initial agent positions:")
    print(penalty_initial)
    print("Baseline landmarks:")
    print(baseline_landmarks)
    print("Penalty landmarks:")
    print(penalty_landmarks)
    print("Initial-state equality: exact element-wise match.")
    limits = axis_limits(
        [baseline_trajectory, penalty_trajectory],
        [baseline_landmarks, penalty_landmarks],
    )
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    draw_trajectory(
        axes[0], baseline_trajectory, baseline_landmarks, baseline_name, limits
    )
    draw_trajectory(
        axes[1], penalty_trajectory, penalty_landmarks, penalty_name, limits
    )
    fig.suptitle("MAPPO Agent Trajectory Comparison")
    fig.tight_layout()
    fig.savefig(output_dir / "trajectory_comparison.png", dpi=300)
    plt.close(fig)
    return True


def main(argv: Sequence[str] | None = None) -> None:
    """Run one model, save its data/figure, and update comparison if possible."""
    args = parse_args(sys.argv[1:] if argv is None else argv)
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    slug = model_slug(args.model_name)
    model_output_dir = output_dir / slug
    if model_output_dir.exists() and any(model_output_dir.iterdir()):
        raise FileExistsError(
            f"Refusing to overwrite existing trajectory results: {model_output_dir}"
        )
    model_output_dir.mkdir(parents=True, exist_ok=True)

    trajectory, landmarks = rollout(args, output_dir)
    write_trajectory(
        model_output_dir / "trajectory.csv",
        args.model_name,
        args.seed,
        trajectory,
    )
    write_landmarks(
        model_output_dir / "landmarks.csv",
        args.model_name,
        args.seed,
        landmarks,
    )
    image_path = output_dir / f"{slug}_trajectory.png"
    save_single_plot(
        image_path,
        trajectory,
        landmarks,
        f"{args.model_name} — Seed {args.seed}",
    )
    comparison_created = maybe_save_comparison(output_dir)
    print(f"Saved trajectory data: {model_output_dir}")
    print(f"Saved figure: {image_path}")
    if comparison_created:
        print(f"Saved comparison: {output_dir / 'trajectory_comparison.png'}")
    else:
        print("Comparison will be created after both Baseline and Penalty runs exist.")


if __name__ == "__main__":
    main()
