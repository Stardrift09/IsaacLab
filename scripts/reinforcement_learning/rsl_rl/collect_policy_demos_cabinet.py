#!/usr/bin/env python3
# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Collect source demonstrations for the cabinet MimicGen pipeline using a trained RSL-RL policy.

The script rolls out the given policy, records every successful episode (drawer opened > 0.35 m)
as an HDF5 demo, and saves them in a format directly compatible with ``annotate_demos.py``.

The HDF5 ``env_name`` is set to ``Isaac-Open-Drawer-Franka-IK-Rel-Mimic-v0`` so that
``annotate_demos.py`` replays the demos in the Mimic env to add subtask boundary signals.

Usage
-----
    cd /home/shaotongchen/workspace_eureka/IsaacLabEureka
    python IsaacLab/scripts/reinforcement_learning/rsl_rl/collect_policy_demos_cabinet.py \\
        --task Isaac-Open-Drawer-Franka-IK-Rel-v0 \\
        --checkpoint /path/to/model.pt \\
        --output_file /tmp/cabinet_src_raw.hdf5 \\
        --num_demos 20 \\
        --headless

Notes
-----
- The policy **must** have been trained on ``Isaac-Open-Drawer-Franka-IK-Rel-v0`` (IK-relative
  7-DoF actions).  Policies trained on the Direct env (joint position control) use a different
  action space and cannot be used with MimicGen.
- Success is detected automatically via a "success" termination term; the built-in
  ``ActionStateRecorderManagerCfg`` exports only successful episodes.
"""

import argparse
import sys

from isaaclab.app import AppLauncher

import cli_args  # isort: skip

parser = argparse.ArgumentParser(description="Collect policy demos for cabinet MimicGen.")
parser.add_argument("--task", type=str, default="Isaac-Open-Drawer-Franka-IK-Rel-v0")
parser.add_argument("--agent", type=str, default="rsl_rl_cfg_entry_point")
parser.add_argument("--output_file", type=str, default="/tmp/cabinet_src_raw.hdf5")
parser.add_argument("--num_demos", type=int, default=20, help="Number of successful demos to collect.")
parser.add_argument("--max_episodes", type=int, default=500, help="Max episodes to attempt.")
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ---------------------------------------------------------------------------
import os

import torch
import gymnasium as gym
from rsl_rl.runners import DistillationRunner, OnPolicyRunner

import isaaclab_tasks  # noqa: F401
import isaaclab_tasks.manager_based.manipulation.cabinet  # noqa: F401
import isaaclab_mimic.envs  # noqa: F401

from isaaclab.envs import ManagerBasedRLEnv, ManagerBasedRLEnvCfg
from isaaclab.envs.mdp.recorders.recorders_cfg import ActionStateRecorderManagerCfg
from isaaclab.managers import DatasetExportMode, TerminationTermCfg as DoneTerm
from isaaclab.utils.assets import retrieve_file_path
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from isaaclab_tasks.utils.hydra import hydra_task_config


def drawer_opened_success(env: ManagerBasedRLEnv, threshold: float = 0.39) -> torch.Tensor:
    """True when the top drawer joint position exceeds *threshold* metres."""
    cabinet = env.scene["cabinet"]
    joint_idx = cabinet.find_joints("drawer_top_joint")[0][0]
    return cabinet.data.joint_pos[:, joint_idx] > threshold


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg, agent_cfg):
    # ---- env config overrides ------------------------------------------------
    env_cfg.scene.num_envs = 1
    env_cfg.sim.device = args_cli.device if args_cli.device else env_cfg.sim.device
    # The HDF5 env_name must point to the *Mimic* env so annotate_demos can replay
    env_cfg.env_name = "Isaac-Open-Drawer-Franka-IK-Rel-Mimic-v0"
    # Disable time-out so it cannot terminate before success; keep only success term.
    # (time_out is harmless to keep if you prefer shorter episodes; remove the line below)
    # env_cfg.terminations.time_out = None

    # Add a "success" termination — the RecorderManager checks this term by name
    # to decide whether to export the episode (EXPORT_SUCCEEDED_ONLY mode).
    env_cfg.terminations.success = DoneTerm(
        func=drawer_opened_success, params={"threshold": 0.39}
    )

    # ---- recorder setup ------------------------------------------------------
    output_dir = os.path.dirname(os.path.abspath(args_cli.output_file))
    output_file_name = os.path.splitext(os.path.basename(args_cli.output_file))[0]
    os.makedirs(output_dir, exist_ok=True)

    env_cfg.recorders = ActionStateRecorderManagerCfg()
    env_cfg.recorders.dataset_export_dir_path = output_dir
    env_cfg.recorders.dataset_filename = output_file_name
    env_cfg.recorders.dataset_export_mode = DatasetExportMode.EXPORT_SUCCEEDED_ONLY

    # ---- create env & policy -------------------------------------------------
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

    # ---- rollout loop --------------------------------------------------------
    obs = env_rsl.get_observations()
    episode_count = 0
    recorder = env.unwrapped.recorder_manager

    print(f"[INFO] Collecting {args_cli.num_demos} successful demos (max {args_cli.max_episodes} episodes)…")

    with torch.inference_mode():
        while recorder.exported_successful_episode_count < args_cli.num_demos:
            if episode_count >= args_cli.max_episodes:
                print(f"[WARN] Reached max_episodes={args_cli.max_episodes}. Stopping early.")
                break

            actions = policy(obs)
            obs, _, dones, _ = env_rsl.step(actions)
            policy_nn.reset(dones)

            if dones[0]:
                episode_count += 1
                n = recorder.exported_successful_episode_count
                print(f"  Episode {episode_count}: {n}/{args_cli.num_demos} demos collected")

    n_final = recorder.exported_successful_episode_count
    print(f"\n[INFO] Collected {n_final} demos → {args_cli.output_file}")
    if n_final < args_cli.num_demos:
        print(f"[WARN] Only {n_final}/{args_cli.num_demos} demos collected. "
              "Consider a better checkpoint or increasing --max_episodes.")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
