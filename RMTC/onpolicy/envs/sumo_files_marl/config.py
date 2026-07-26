#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Default configuration for the SUMO multi-agent traffic environment.

This module provides a ``config`` object that matches the shape expected by
``train_sumo.py`` and ``SUMO_env.py``.
"""

import os


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_ROOT = os.path.join(BASE_DIR, "results", "sumo_logs") + os.sep

config = {
    "episode": {
        "num_train_rollouts": 100000,
        "rollout_length": 240,
        "warmup_ep_steps": 0,
        "test_num_eps": 50,
    },
    "environment": {
        "name": "sumo_marl",
        "agent": "",
        "action_type": "select_phase",
        "gui": False,
        "yellow_duration": 5,
        "iter_duration": 10,
        "episode_length_time": 3600,
        "is_record": False,
        "is_libsumo": True,
        "output_path": OUTPUT_ROOT,
        "num_actions": 8,
        "obs_shape": 72,
        "vehicle_num_actions": 8,
        "vehicle_obs_shape": 8,
        "sumocfg_files": [
            "sumo_files_marl/scenarios/fenglin/fenglin/sumo_fenglin_base_road_trip/base.sumocfg"
        ],
        "reward_type": ["queue_len", "wait_time", "delay_time", "pressure", "speed_score"],
        "reward_type_ve": ["queue_len", "wait_time", "delay_time", "pressure", "speed_score"],
        "reward_type_emv": ["queue_len", "wait_time", "delay_time", "pressure", "speed_score"],
        "state_key": [
            "current_phase",
            "car_num",
            "queue_length",
            "occupancy",
            "flow",
            "stop_car_num",
            "pressure",
        ],
        "ve_state_key": [
            "current_intersection",
            "current_intersection_direction",
            "destination_intersection",
            "destination_intersection_direction",
        ],
    },
}


__all__ = ["config"]
