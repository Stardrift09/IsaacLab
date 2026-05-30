# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Replay a single trajectory of an RSL-RL checkpoint and run VLM feedback on it.

This is a thin variant of play.py that:
  * Forces the environment's `camera_sensor_record` flag on, so the env defines
    its task-specific camera inside `_setup_scene`.
  * Skips video recording and JIT/ONNX export.
  * After loading the policy, calls
    `env.unwrapped.run_single_traj_and_get_vlm_feedback(policy_nn, policy, output_dir)`
    which rolls out one episode in env 0, dumps RGB frames, and (unless
    `--skip_vlm`) runs the VLM and prints its response.
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import os
import sys
import time

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Replay a trajectory and run VLM feedback.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments (camera attaches to env 0).")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the pre-trained checkpoint from Nucleus.",
)
parser.add_argument(
    "--output_dir",
    type=str,
    default=None,
    help="Directory to write RGB frames into. Defaults to <log_dir>/vlm_frames/<timestamp>.",
)
parser.add_argument(
    "--skip_vlm",
    action="store_true",
    default=False,
    help="Roll out and save frames but do NOT load the VLM (faster, no model download).",
)
parser.add_argument(
    "--stochastic",
    action="store_true",
    default=False,
    help="Use sampled actions instead of mean (matches training behaviour).",
)
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()
# camera-only run: do not need offscreen video pipeline, but we DO need cameras enabled
args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import torch
from rsl_rl.runners import DistillationRunner, OnPolicyRunner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.assets import retrieve_file_path

from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper
from isaaclab_rl.utils.pretrained_checkpoint import get_published_pretrained_checkpoint

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

# PLACEHOLDER: Extension template (do not remove this comment)


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    """Replay one trajectory and run VLM feedback."""
    task_name = args_cli.task.split(":")[-1]
    train_task_name = task_name.replace("-Play", "")

    agent_cfg: RslRlBaseRunnerCfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs

    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # Force the env to define its task-specific camera inside _setup_scene.
    if not hasattr(env_cfg, "camera_sensor_record"):
        raise AttributeError(
            f"env_cfg for task '{args_cli.task}' has no 'camera_sensor_record' attribute; "
            "this script expects an env that defines its own camera (see test_place_basket_in_drawer.py)."
        )
    env_cfg.camera_sensor_record = True

    # locate checkpoint
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    if args_cli.use_pretrained_checkpoint:
        resume_path = get_published_pretrained_checkpoint("rsl_rl", train_task_name)
        if not resume_path:
            print("[INFO] No pre-trained checkpoint available for this task.")
            return
    elif args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    log_dir = os.path.dirname(resume_path)
    env_cfg.log_dir = log_dir

    # decide output directory for RGB frames
    if args_cli.output_dir is not None:
        output_dir = os.path.abspath(args_cli.output_dir)
    else:
        output_dir = os.path.join(log_dir, "vlm_frames", time.strftime("%Y-%m-%d_%H-%M-%S"))
    os.makedirs(output_dir, exist_ok=True)
    print(f"[INFO] Writing RGB frames to: {output_dir}")

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)

    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    print(f"[INFO] Loading model checkpoint from: {resume_path}")
    if agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    elif agent_cfg.class_name == "DistillationRunner":
        runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
    runner.load(resume_path, map_location=agent_cfg.device)

    if args_cli.stochastic:
        policy = lambda obs: runner.alg.policy.act(obs)
    else:
        policy = runner.get_inference_policy(device=env.unwrapped.device)

    try:
        # version 2.3 onwards
        policy_nn = runner.alg.policy
    except AttributeError:
        # version 2.2 and below
        policy_nn = runner.alg.actor_critic

    # Run one trajectory + collect frames (+ optionally VLM feedback)
    direct_env = env.unwrapped
    if not hasattr(direct_env, "run_single_traj_and_get_vlm_feedback"):
        raise AttributeError(
            f"env class {type(direct_env).__name__} does not implement "
            "run_single_traj_and_get_vlm_feedback(policy_nn, policy, output_dir)."
        )

    with torch.inference_mode():
        if args_cli.skip_vlm:
            # Replicate the rollout-and-save part without invoking the VLM.
            # We do this by monkey-patching get_feedback_from_vlm for this call.
            original = getattr(direct_env, "get_feedback_from_vlm", None)
            direct_env.get_feedback_from_vlm = lambda output_dir: "[skipped: --skip_vlm set]"
            try:
                feedback = direct_env.run_single_traj_and_get_vlm_feedback(policy_nn, policy, output_dir)
            finally:
                if original is not None:
                    direct_env.get_feedback_from_vlm = original
        else:
            feedback = direct_env.run_single_traj_and_get_vlm_feedback(policy_nn, policy, output_dir)

    print("=" * 80)
    print("[VLM FEEDBACK]")
    print(feedback)
    print("=" * 80)

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
