#!/usr/bin/env python3
# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Hybrid play script: motion planner approaches drawer handle, then RL policy opens it.

Phase 1 (approach): simple proportional controller moves EEF toward the drawer handle
         using the IK-rel action space.  Generalises to any cabinet position without
         any IL data.
Phase 2 (open):    trained RL policy takes over once EEF is within --approach_dist of
         the handle.

Usage
-----
    cd /home/shaotongchen/workspace_eureka/IsaacLabEureka
    python IsaacLab/scripts/reinforcement_learning/rsl_rl/play_cabinet_hybrid.py \\
        --task Isaac-Open-Drawer-Franka-IK-Rel-v0 \\
        --checkpoint /path/to/model.pt \\
        --num_envs 4 \\
        --approach_dist 0.12 \\
        --headless
"""

import argparse
import sys

from isaaclab.app import AppLauncher

import cli_args  # isort: skip

parser = argparse.ArgumentParser(description="Hybrid motion-plan + RL policy for cabinet opening.")
parser.add_argument("--task", type=str, default="Isaac-Open-Drawer-Franka-IK-Rel-v0")
parser.add_argument("--agent", type=str, default="rsl_rl_cfg_entry_point")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--approach_dist", type=float, default=0.12,
                    help="Switch from motion planner to policy when EEF-handle dist < this (m).")
parser.add_argument("--approach_speed", type=float, default=0.06,
                    help="Motion planner step size in metres per env step.")
parser.add_argument("--max_steps", type=int, default=500,
                    help="Max steps per episode before forced reset.")
parser.add_argument("--num_episodes", type=int, default=20,
                    help="Number of episodes to run.")
parser.add_argument("--real-time", action="store_true", default=False)
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ---------------------------------------------------------------------------
import time

import gymnasium as gym
import torch
from rsl_rl.runners import DistillationRunner, OnPolicyRunner

import isaaclab_tasks  # noqa: F401
import isaaclab_tasks.manager_based.manipulation.cabinet  # noqa: F401

from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.utils.assets import retrieve_file_path

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from isaaclab_tasks.utils.hydra import hydra_task_config


def motion_plan_action(env_unwrapped, approach_speed: float, ik_scale: float = 0.5) -> torch.Tensor:
    """Proportional controller: move EEF straight toward the drawer handle.

    Returns action tensor shaped (N, action_dim) in the IK-rel action space.
    Pos dims are filled with a clamped step toward the handle; rot dims are zero
    (keep current orientation); gripper is open.
    """
    ee_pos = env_unwrapped.scene["ee_frame"].data.target_pos_w[:, 0, :]      # (N, 3)
    handle_pos = env_unwrapped.scene["cabinet_frame"].data.target_pos_w[:, 0, :]  # (N, 3)

    delta = handle_pos - ee_pos                                               # (N, 3)
    dist = torch.linalg.norm(delta, dim=-1, keepdim=True).clamp(min=1e-6)
    direction = delta / dist                                                  # unit vector

    # Convert world-space step to action value:  action * ik_scale = actual delta
    step = direction * approach_speed
    action_pos = (step / ik_scale).clamp(-1.0, 1.0)                          # (N, 3)

    N = ee_pos.shape[0]
    device = ee_pos.device
    action_rot = torch.zeros(N, 3, device=device)
    action_gripper = torch.ones(N, 1, device=device)                         # open

    return torch.cat([action_pos, action_rot, action_gripper], dim=-1)       # (N, 7)


def get_ee_handle_dist(env_unwrapped) -> torch.Tensor:
    """Returns (N,) tensor of EEF-to-handle distances in world space."""
    ee_pos = env_unwrapped.scene["ee_frame"].data.target_pos_w[:, 0, :]
    handle_pos = env_unwrapped.scene["cabinet_frame"].data.target_pos_w[:, 0, :]
    return torch.linalg.norm(handle_pos - ee_pos, dim=-1)


def get_drawer_pos(env_unwrapped) -> torch.Tensor:
    """Returns (N,) tensor of drawer_top_joint positions."""
    cabinet = env_unwrapped.scene["cabinet"]
    joint_idx = cabinet.find_joints("drawer_top_joint")[0][0]
    return cabinet.data.joint_pos[:, joint_idx]


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg, agent_cfg):
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device else env_cfg.sim.device

    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)

    env = gym.make(args_cli.task, cfg=env_cfg)
    env_rsl = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    checkpoint_path = retrieve_file_path(args_cli.checkpoint)
    print(f"[INFO] Loading checkpoint: {checkpoint_path}")

    if agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(env_rsl, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    elif agent_cfg.class_name == "DistillationRunner":
        runner = DistillationRunner(env_rsl, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")

    runner.load(checkpoint_path, map_location=agent_cfg.device)
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    try:
        policy_nn = runner.alg.policy
    except AttributeError:
        policy_nn = runner.alg.actor_critic

    N = args_cli.num_envs
    device = env.unwrapped.device
    success_threshold = 0.39

    # Track per-env state
    using_policy = torch.zeros(N, dtype=torch.bool, device=device)  # True = policy phase
    step_counts = torch.zeros(N, dtype=torch.long, device=device)

    episode_num = 0
    successes = 0

    obs = env_rsl.get_observations()
    dt = env.unwrapped.step_dt

    print(f"\n[INFO] Hybrid play: approach_dist={args_cli.approach_dist}m  "
          f"approach_speed={args_cli.approach_speed}m/step\n")

    with torch.inference_mode():
        while episode_num < args_cli.num_episodes and simulation_app.is_running():
            start_time = time.time()

            # Check which envs are close enough to switch to policy
            dists = get_ee_handle_dist(env.unwrapped)
            newly_switched = (~using_policy) & (dists < args_cli.approach_dist)
            if newly_switched.any():
                print(f"  [step {step_counts[0].item():4d}] Switching to policy — "
                      f"dist={dists[0].item():.3f}m")
            using_policy |= newly_switched

            # Build actions: motion plan or policy per env
            if using_policy.all():
                actions = policy(obs)
            elif using_policy.any():
                # Mixed: build both and select per env
                mp_actions = motion_plan_action(env.unwrapped, args_cli.approach_speed)
                rl_actions = policy(obs)
                mask = using_policy.unsqueeze(1).float()
                actions = rl_actions * mask + mp_actions * (1.0 - mask)
            else:
                actions = motion_plan_action(env.unwrapped, args_cli.approach_speed)

            obs, _, dones, extras = env_rsl.step(actions)
            policy_nn.reset(dones)
            step_counts += 1

            # time_outs[i]=True  → episode ended by timeout (NOT success)
            # time_outs[i]=False → episode ended by non-timeout termination (= success)
            time_outs = extras.get("time_outs", torch.zeros(N, dtype=torch.bool, device=device))

            # Handle env-driven episode ends (success or built-in timeout)
            for i in range(N):
                if dones[i]:
                    success = not time_outs[i].item()
                    if success:
                        successes += 1
                    episode_num += 1
                    phase = "policy" if using_policy[i].item() else "approach"
                    print(f"  Episode {episode_num:3d} env{i}: "
                          f"{'SUCCESS' if success else 'timeout'}  "
                          f"steps={step_counts[i].item()}  phase={phase}")
                    using_policy[i] = False
                    step_counts[i] = 0

            # Handle our manual step-cap (counts as timeout, not success)
            for i in range(N):
                if not dones[i] and step_counts[i] >= args_cli.max_steps:
                    drawer_now = get_drawer_pos(env.unwrapped)[i].item()
                    episode_num += 1
                    phase = "policy" if using_policy[i].item() else "approach"
                    print(f"  Episode {episode_num:3d} env{i}: "
                          f"max_steps  drawer={drawer_now:.3f}m  "
                          f"steps={step_counts[i].item()}  phase={phase}")
                    using_policy[i] = False
                    step_counts[i] = 0

            if args_cli.real_time:
                sleep_time = dt - (time.time() - start_time)
                if sleep_time > 0:
                    time.sleep(sleep_time)

    print(f"\n[INFO] Results: {successes}/{episode_num} episodes succeeded "
          f"({100*successes/max(episode_num,1):.1f}%)")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
