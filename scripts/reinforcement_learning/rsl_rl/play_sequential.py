# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play two RSL-RL checkpoints sequentially with rule-based policy switching.

Loads a PickItUp policy and a PlaceInBasket policy. Uses the PlaceInBasket env
(start_in_air=False) so the episode doesn't terminate on pick success.

Switching rule:
  - Phase 0 (pick): run pick policy until grasped & high_enough & small_rotation
  - Phase 1 (place): run place policy until episode terminates (inside_site & low_enough)
  - On episode done: reset to phase 0
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Sequential policy play with rule-based switching.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during play.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument(
    "--task",
    type=str,
    default="PlaceInBasket",
    help="Gym task name for the environment (default: PlaceInBasket — survives pick phase).",
)
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment.")
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
parser.add_argument("--stochastic", action="store_true", default=False, help="Use sampled actions instead of mean (matches training behavior).")
parser.add_argument(
    "--pick_checkpoint",
    type=str,
    required=True,
    help="Path to the PickItUp policy checkpoint (.pt).",
)
parser.add_argument(
    "--place_checkpoint",
    type=str,
    required=True,
    help="Path to the PlaceInBasket policy checkpoint (.pt).",
)
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import os
import time

import gymnasium as gym
import torch
from rsl_rl.runners import OnPolicyRunner

from isaaclab.envs import DirectMARLEnv, DirectMARLEnvCfg, DirectRLEnvCfg, ManagerBasedRLEnvCfg, multi_agent_to_single_agent
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict

from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.hydra import hydra_task_config


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    """Run pick + place policies sequentially with rule-based switching."""

    # override configurations with non-hydra CLI arguments
    agent_cfg: RslRlBaseRunnerCfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # Force start_in_air=False so the env runs the full pick→place pipeline from ground
    env_cfg.start_in_air = False

    # resolve checkpoint paths
    pick_resume_path = retrieve_file_path(args_cli.pick_checkpoint)
    place_resume_path = retrieve_file_path(args_cli.place_checkpoint)
    print(f"[INFO] Pick checkpoint:  {pick_resume_path}")
    print(f"[INFO] Place checkpoint: {place_resume_path}")

    # derive log dir from pick checkpoint (used for video output)
    log_dir = os.path.dirname(pick_resume_path)
    env_cfg.log_dir = log_dir

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play_sequential"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during play.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap for rsl-rl
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    device = env.unwrapped.device

    # --- Load pick policy ---
    pick_runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    pick_runner.load(pick_resume_path, map_location=agent_cfg.device)
    try:
        pick_policy_nn = pick_runner.alg.policy
    except AttributeError:
        pick_policy_nn = pick_runner.alg.actor_critic
    if args_cli.stochastic:
        pick_policy = lambda obs: pick_policy_nn.act(obs)
    else:
        pick_policy = pick_runner.get_inference_policy(device=device)

    # --- Load place policy (shares same env object; runners only read obs/action dims) ---
    place_runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    place_runner.load(place_resume_path, map_location=agent_cfg.device)
    try:
        place_policy_nn = place_runner.alg.policy
    except AttributeError:
        place_policy_nn = place_runner.alg.actor_critic
    if args_cli.stochastic:
        place_policy = lambda obs: place_policy_nn.act(obs)
    else:
        place_policy = place_runner.get_inference_policy(device=device)

    dt = env.unwrapped.step_dt

    # per-env phase: 0 = pick, 1 = place
    phase = torch.zeros(env.num_envs, dtype=torch.long, device=device)

    # reset environment
    obs = env.get_observations()
    timestep = 0

    print("[INFO] Starting sequential play. Phase 0 = pick, Phase 1 = place.")

    while simulation_app.is_running():
        start_time = time.time()

        with torch.inference_mode():
            # evaluate both policies on the full batch
            pick_actions = pick_policy(obs)
            place_actions = place_policy(obs)

            # select per-env action based on current phase
            actions = torch.where(phase.unsqueeze(-1) == 0, pick_actions, place_actions)

            obs, _, dones, _ = env.step(actions)

        # --- rule-based phase update (outside inference_mode for in-place ops) ---
        env_core = env.unwrapped
        transition = env_core.grasped & env_core.high_enough
        phase[(phase == 0) & transition] = 1   # pick → place: object lifted and grasped
        phase[dones.bool()] = 0                # episode done → reset to pick
        print(phase)
        # reset recurrent hidden states for terminated episodes
        pick_policy_nn.reset(dones)
        place_policy_nn.reset(dones)

        if args_cli.video:
            timestep += 1
            if timestep == args_cli.video_length:
                break

        # time delay for real-time evaluation
        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
