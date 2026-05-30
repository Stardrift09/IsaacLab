"""Generate source HDF5 demonstrations for the Franka pick-basket task using a scripted controller.

The script runs a deterministic state-machine controller that:
    State 0  approach_above  — move EEF to [obj_xy, obj_z + approach_height], open gripper
    State 1  descend         — move EEF down to grasp height above object
    State 2  grasp           — hold pose, close gripper
    State 3  lift            — move straight up
    State 4  move_over       — move EEF over basket
    State 5  release         — open gripper

Each successful episode is recorded to an HDF5 file compatible with
``isaaclab_mimic/scripts/annotate_demos.py``.

Usage::

    python scripts/tools/generate_scripted_demos_pick_basket.py \\
        --output_file /tmp/pick_basket_src_demos.hdf5 \\
        --num_demos 5 \\
        --headless

After generation run annotation::

    python IsaacLab/scripts/imitation_learning/isaaclab_mimic/annotate_demos.py \\
        --input_file /tmp/pick_basket_src_demos.hdf5 \\
        --env_id Isaac-Franka-PickBasket-IK-Rel-Mimic-v0
"""

import argparse

import torch

# IsaacLab app launcher must be imported first
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Generate scripted demonstrations for Franka pick-basket")
parser.add_argument("--output_file", type=str, default="/tmp/pick_basket_src_demos.hdf5")
parser.add_argument("--num_demos", type=int, default=5, help="Number of demos to record")
parser.add_argument("--num_envs", type=int, default=1, help="Number of parallel envs (use 1 for scripted)")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

# ---- imports after sim launch ----
import gymnasium as gym
import h5py
import numpy as np

import isaaclab.utils.math as math_utils

# Register the task
import isaaclab_tasks.manager_based.manipulation.pick_basket.config.franka  # noqa: F401
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg


# ------------------------------------------------------------------
# State machine controller
# ------------------------------------------------------------------

class PickBasketStateMachine:
    """Deterministic scripted controller for pick-basket.

    Actions are delta-EEF poses (6-DoF) + binary gripper (1 scalar).
    Compatible with ``DifferentialInverseKinematicsActionCfg`` in relative mode.
    """

    # Steps per state
    STEPS = {
        "approach_above": 60,
        "descend": 40,
        "grasp": 25,
        "lift": 40,
        "move_over": 50,
        "release": 20,
        "done": 1,
    }
    STATES = list(STEPS.keys())

    # EEF control gains (position delta per step, m)
    POS_STEP = 0.015
    # Gripper values
    GRIPPER_OPEN = 1.0    # BinaryJointPositionAction: 1 = open
    GRIPPER_CLOSE = -1.0  # -1 = close

    def __init__(self, device: str):
        self.device = device
        self._state_idx = 0
        self._step_in_state = 0

    @property
    def state(self) -> str:
        return self.STATES[self._state_idx]

    def reset(self):
        self._state_idx = 0
        self._step_in_state = 0

    def is_done(self) -> bool:
        return self.state == "done"

    def compute_action(
        self,
        eef_pos: torch.Tensor,      # (3,) current EEF position (env-local)
        obj_pos: torch.Tensor,       # (3,) object position (env-local)
        basket_pos: torch.Tensor,    # (3,) basket position (env-local)
        approach_height: float = 0.17,
        grasp_height_above_obj: float = 0.02,
        lift_height: float = 0.25,
        over_basket_height: float = 0.28,
    ) -> torch.Tensor:
        """Return 7-DoF action: [dx, dy, dz, dax, day, daz, gripper]."""
        action = torch.zeros(7, device=self.device)

        if self.state == "approach_above":
            target = obj_pos.clone()
            target[2] = approach_height
            action[:3] = self._step_toward(eef_pos, target)
            action[6] = self.GRIPPER_OPEN

        elif self.state == "descend":
            target = obj_pos.clone()
            target[2] = obj_pos[2] + grasp_height_above_obj
            action[:3] = self._step_toward(eef_pos, target)
            action[6] = self.GRIPPER_OPEN

        elif self.state == "grasp":
            # Hold position, close gripper
            action[6] = self.GRIPPER_CLOSE

        elif self.state == "lift":
            target = eef_pos.clone()
            target[2] = lift_height
            action[:3] = self._step_toward(eef_pos, target)
            action[6] = self.GRIPPER_CLOSE

        elif self.state == "move_over":
            target = basket_pos.clone()
            target[2] = over_basket_height
            action[:3] = self._step_toward(eef_pos, target)
            action[6] = self.GRIPPER_CLOSE

        elif self.state == "release":
            action[6] = self.GRIPPER_OPEN

        # Advance state
        self._step_in_state += 1
        if self._step_in_state >= self.STEPS[self.state]:
            self._state_idx = min(self._state_idx + 1, len(self.STATES) - 1)
            self._step_in_state = 0

        return action

    def _step_toward(self, current: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Clipped step toward target position."""
        delta = target - current
        norm = torch.linalg.norm(delta)
        if norm < 1e-6:
            return torch.zeros_like(delta)
        step = min(self.POS_STEP, norm.item())
        return (delta / norm) * step


# ------------------------------------------------------------------
# Demo recording helpers
# ------------------------------------------------------------------

def record_demo(env, controller: PickBasketStateMachine, max_steps: int = 300):
    """Run one episode and return trajectory dict or None on failure."""
    obs, _ = env.reset()
    controller.reset()

    traj = {"obs": [], "actions": [], "states": []}
    success = False

    for _ in range(max_steps):
        eef_pos = obs["policy"]["eef_pos"][0].cpu()       # (3,)
        obj_pos = obs["policy"]["object_pos"][0].cpu()    # (3,)
        basket_pos = obs["policy"]["basket_pos"][0].cpu() # (3,)

        action = controller.compute_action(eef_pos, obj_pos, basket_pos)
        action_batch = action.unsqueeze(0).to(env.device)

        traj["obs"].append({k: v[0].cpu().numpy() for k, v in obs["policy"].items()})
        traj["actions"].append(action.cpu().numpy())
        # scene state for mimic annotation
        traj["states"].append(env.scene.get_state(is_relative=False))

        obs, _, terminated, truncated, info = env.step(action_batch)

        if terminated[0].item():
            success = True
            break
        if truncated[0].item() or controller.is_done():
            break

    return traj if success else None


def save_demos(demos: list, output_file: str):
    with h5py.File(output_file, "w") as f:
        data_grp = f.create_group("data")
        for i, demo in enumerate(demos):
            ep_grp = data_grp.create_group(f"demo_{i}")
            # Observations
            obs_grp = ep_grp.create_group("obs")
            for key in demo["obs"][0].keys():
                arr = np.stack([o[key] for o in demo["obs"]])
                obs_grp.create_dataset(key, data=arr)
            # Actions
            ep_grp.create_dataset("actions", data=np.stack(demo["actions"]))
        data_grp.attrs["num_demos"] = len(demos)
        data_grp.attrs["env_name"] = "Isaac-Franka-PickBasket-IK-Rel-Mimic-v0"
    print(f"Saved {len(demos)} demos to {output_file}")


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main():
    env_cfg = parse_env_cfg("Isaac-Franka-PickBasket-IK-Rel-v0", device=args.device, num_envs=args.num_envs)
    env = gym.make("Isaac-Franka-PickBasket-IK-Rel-v0", cfg=env_cfg)
    controller = PickBasketStateMachine(device=env.unwrapped.device)

    demos = []
    attempts = 0
    max_attempts = args.num_demos * 10

    print(f"Recording {args.num_demos} demonstrations...")
    while len(demos) < args.num_demos and attempts < max_attempts:
        attempts += 1
        demo = record_demo(env, controller)
        if demo is not None:
            demos.append(demo)
            print(f"  [{len(demos)}/{args.num_demos}] success (attempt {attempts})")
        else:
            print(f"  attempt {attempts}: failed, retrying...")

    env.close()

    if len(demos) == 0:
        print("ERROR: No successful demos recorded. Check state machine parameters.")
        return

    save_demos(demos, args.output_file)
    print(f"\nDone. Run annotation next:")
    print(
        f"  python IsaacLab/scripts/imitation_learning/isaaclab_mimic/annotate_demos.py"
        f" --input_file {args.output_file}"
        f" --env_id Isaac-Franka-PickBasket-IK-Rel-Mimic-v0"
    )


if __name__ == "__main__":
    main()
    simulation_app.close()
