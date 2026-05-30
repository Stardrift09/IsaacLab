"""Convert LIBERO pick-basket demonstrations to IsaacLab MimicGen HDF5 format.

Replays each LIBERO episode in the IsaacLab env using direct joint position
control, records the resulting EEF delta actions, and saves in the format
expected by annotate_demos.py.

Supports:
  libero/trajs/libero/pick_up_the_alphabet_soup_and_place_it_in_the_basket/v2/franka_v2.pkl.gz
  libero/trajs/libero90/libero_90_living_room_scene1_pick_up_the_alphabet_soup_and_put_it_in_the_basket_traj_v2.pkl

Coordinate convention
---------------------
LIBERO stores positions in its own world frame with the robot base at some
offset. We convert to robot-relative coordinates by subtracting the LIBERO
robot world position. In IsaacLab the robot sits at the env origin, so the
robot-relative offset is placed directly relative to env_origins.

Quaternion conventions
----------------------
The v2 pkl files already store quaternions in IsaacLab wxyz format (confirmed:
franka identity rot=[1,0,0,0] = w=1).  No conversion needed.

Action format
-------------
The env uses DifferentialInverseKinematicsActionCfg(scale=0.5, relative mode)
+ BinaryJointPositionActionCfg.  A 7-D action is recorded per step:
  [dx, dy, dz, dax, day, daz, gripper]
where (dx,dy,dz) = EEF position delta / 0.5 and (dax,day,daz) is the
axis-angle rotation delta / 0.5.  Gripper: +1=open, -1=close.

Usage::

    python IsaacLab/scripts/tools/libero_to_isaaclab_demos.py \\
        --libero_pkl libero/trajs/libero/pick_up_the_alphabet_soup_and_place_it_in_the_basket/v2/franka_v2.pkl.gz \\
        --output_file /tmp/pick_basket_src_demos.hdf5 \\
        --num_demos 10 \\
        --headless

Then run annotation::

    python IsaacLab/scripts/imitation_learning/isaaclab_mimic/annotate_demos.py \\
        --input_file /tmp/pick_basket_src_demos.hdf5 \\
        --output_file /tmp/pick_basket_src_annotated.hdf5 \\
        --env_id Isaac-Franka-PickBasket-IK-Rel-Mimic-v0 \\
        --auto \\
        --headless
"""

import argparse
import gzip
import pickle

import torch

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Convert LIBERO demos to IsaacLab HDF5 format")
parser.add_argument(
    "--libero_pkl",
    type=str,
    default="libero/trajs/libero/pick_up_the_alphabet_soup_and_place_it_in_the_basket/v2/franka_v2.pkl.gz",
    help="Path to LIBERO trajectory .pkl or .pkl.gz file",
)
parser.add_argument("--output_file", type=str, default="/tmp/pick_basket_src_demos.hdf5")
parser.add_argument("--num_demos", type=int, default=10, help="Max demos to convert")
parser.add_argument(
    "--start_idx",
    type=int,
    default=0,
    help="Start episode index in the LIBERO pkl (useful for selecting diverse episodes)",
)
parser.add_argument("--stabilize_steps", type=int, default=5, help="Extra sim steps after state reset to stabilize")
parser.add_argument("--gripper_threshold", type=float, default=0.02, help="Finger joint threshold for open/close")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

# ---- imports after sim launch ----
import h5py
import numpy as np
import gymnasium as gym

import isaaclab.utils.math as math_utils

import isaaclab_tasks.manager_based.manipulation.pick_basket.config.franka  # noqa: F401
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

# LIBERO joint name order (used to build ordered tensor)
LIBERO_JOINT_NAMES = [
    "panda_joint1", "panda_joint2", "panda_joint3", "panda_joint4",
    "panda_joint5", "panda_joint6", "panda_joint7",
    "panda_finger_joint1", "panda_finger_joint2",
]
ARM_DOF = 7   # first 7 joints
FINGER_DOF = 2
TOTAL_DOF = ARM_DOF + FINGER_DOF

IK_SCALE = 0.5   # matches DifferentialInverseKinematicsActionCfg(scale=0.5)


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def load_libero_pkl(path: str) -> list[dict]:
    """Load LIBERO trajectory file (.pkl or .pkl.gz), return list of episodes."""
    opener = gzip.open if path.endswith(".gz") else open
    mode = "rb"
    with opener(path, mode) as f:
        data = pickle.load(f)
    # Both formats have {"franka": [ep0, ep1, ...]}
    return data["franka"]


def libero_dof_pos_to_tensor(dof_dict: dict, device: str) -> torch.Tensor:
    """Convert LIBERO dof_pos dict to ordered tensor of shape (9,)."""
    return torch.tensor(
        [dof_dict[name] for name in LIBERO_JOINT_NAMES], dtype=torch.float32, device=device
    )


def compute_delta_eef_action(
    pos_prev: torch.Tensor,
    quat_prev: torch.Tensor,
    pos_curr: torch.Tensor,
    quat_curr: torch.Tensor,
    finger_pos: torch.Tensor,
    gripper_threshold: float,
) -> torch.Tensor:
    """Compute 7-D delta EEF action from consecutive EEF poses.

    All positions / quaternions must be in the same frame (robot base frame).
    Quaternions in IsaacLab convention [w, x, y, z].
    """
    # Position delta (divided by IK scale)
    delta_pos = (pos_curr - pos_prev) / IK_SCALE

    # Rotation delta: q_rel = q_curr * q_prev^{-1}
    q_prev_inv = math_utils.quat_inv(quat_prev.unsqueeze(0)).squeeze(0)
    q_rel = math_utils.quat_mul(quat_curr.unsqueeze(0), q_prev_inv.unsqueeze(0)).squeeze(0)
    delta_rot = math_utils.axis_angle_from_quat(q_rel.unsqueeze(0)).squeeze(0) / IK_SCALE

    # Gripper: +1 = open, -1 = close
    gripper = torch.tensor(
        [1.0 if finger_pos.mean().item() > gripper_threshold else -1.0],
        device=pos_curr.device,
    )

    return torch.cat([delta_pos, delta_rot, gripper], dim=0)  # (7,)


def write_hdf5(output_file: str, env_name: str, demos: list[dict]):
    """Write demos to HDF5 compatible with HDF5DatasetFileHandler / annotate_demos.py.

    Each demo dict has:
        "initial_state": nested dict matching scene.get_state(is_relative=True)
        "actions":       np.ndarray of shape (T, 7)
    """
    import json

    with h5py.File(output_file, "w") as f:
        data_grp = f.create_group("data")
        data_grp.attrs["env_args"] = json.dumps({"env_name": env_name, "type": 2})
        data_grp.attrs["total"] = len(demos)

        for i, demo in enumerate(demos):
            ep_grp = data_grp.create_group(f"demo_{i}")

            # --- actions ---
            ep_grp.create_dataset("actions", data=demo["actions"])

            # --- initial_state (nested) ---
            state = demo["initial_state"]

            def write_nested(parent_grp, d):
                for key, val in d.items():
                    if isinstance(val, dict):
                        sub = parent_grp.create_group(key)
                        write_nested(sub, val)
                    else:
                        arr = val.cpu().numpy() if hasattr(val, "cpu") else np.array(val)
                        parent_grp.create_dataset(key, data=arr)

            state_grp = ep_grp.create_group("initial_state")
            write_nested(state_grp, state)

    print(f"Saved {len(demos)} demos to {output_file}")


# ------------------------------------------------------------------
# Main replay loop
# ------------------------------------------------------------------

def main():
    episodes = load_libero_pkl(args.libero_pkl)
    print(f"Loaded {len(episodes)} LIBERO episodes from {args.libero_pkl}")

    env_cfg = parse_env_cfg("Isaac-Franka-PickBasket-IK-Rel-v0", device=args.device, num_envs=1)
    env = gym.make("Isaac-Franka-PickBasket-IK-Rel-v0", cfg=env_cfg)
    env_unwrapped = env.unwrapped
    device = env_unwrapped.device
    scene = env_unwrapped.scene

    robot = scene["robot"]
    soup = scene["alphabet_soup"]
    basket_obj = scene["basket"]
    ee_frame = scene["ee_frame"]

    # Get joint index mapping: IsaacLab joint order → LIBERO joint order
    isaaclab_joint_names = robot.data.joint_names
    libero_to_il_idx = []
    for libero_name in LIBERO_JOINT_NAMES:
        idx = isaaclab_joint_names.index(libero_name)
        libero_to_il_idx.append(idx)
    libero_to_il_idx = torch.tensor(libero_to_il_idx, device=device)

    print(f"Joint mapping verified. IsaacLab joints: {isaaclab_joint_names}")

    # Joint ids for arm only (for set_joint_position_target)
    arm_joint_ids = libero_to_il_idx[:ARM_DOF]
    finger_joint_ids = libero_to_il_idx[ARM_DOF:]

    demos = []
    episode_slice = episodes[args.start_idx: args.start_idx + args.num_demos * 5]  # allow retries

    for ep_idx, ep in enumerate(episode_slice):
        if len(demos) >= args.num_demos:
            break

        print(f"\nReplaying LIBERO episode {args.start_idx + ep_idx} ({len(demos)}/{args.num_demos} collected)...")

        init = ep["init_state"]
        actions_libero = ep["actions"]

        # --- Build initial state in IsaacLab format ---
        # Robot-relative positions from LIBERO world coords
        robot_libero_pos = np.array(init["franka"]["pos"])

        soup_libero_pos = np.array(init["alphabet_soup"]["pos"])
        soup_rel = soup_libero_pos - robot_libero_pos
        soup_quat_wxyz = init["alphabet_soup"]["rot"]  # already wxyz in v2 pkl

        basket_libero_pos = np.array(init["basket"]["pos"])
        basket_rel = basket_libero_pos - robot_libero_pos
        basket_quat_wxyz = init["basket"]["rot"]  # already wxyz in v2 pkl

        robot_init_joints = libero_dof_pos_to_tensor(init["franka"]["dof_pos"], device)

        # Reset env (triggers events etc.)
        env.reset()
        env_unwrapped.sim.step(render=False)
        env_unwrapped.scene.update(env_unwrapped.physics_dt)

        # Get env origin (robot sits here)
        env_origin = scene.env_origins[0]  # (3,)

        # Override scene state with LIBERO initial conditions

        # Robot: set joint positions and zero velocities
        joint_pos_full = torch.zeros(1, len(isaaclab_joint_names), device=device)
        joint_pos_full[0, libero_to_il_idx] = robot_init_joints
        joint_vel_full = torch.zeros_like(joint_pos_full)
        robot.write_joint_state_to_sim(joint_pos_full, joint_vel_full)
        # Set PD targets to match so the actuator doesn't fight the initial state
        robot.set_joint_position_target(joint_pos_full)
        # Robot root at env origin, identity rotation
        robot_root_pose = torch.zeros(1, 7, device=device)
        robot_root_pose[0, 3] = 1.0  # w=1 quaternion
        robot_root_pose[0, :3] = env_origin  # world position
        robot.write_root_pose_to_sim(robot_root_pose)
        robot.write_root_velocity_to_sim(torch.zeros(1, 6, device=device))

        # Alphabet soup
        soup_pos_w = env_origin + torch.tensor(soup_rel, dtype=torch.float32, device=device)
        soup_quat_t = torch.tensor(soup_quat_wxyz, dtype=torch.float32, device=device)
        soup_pose = torch.cat([soup_pos_w, soup_quat_t]).unsqueeze(0)  # (1, 7)
        soup.write_root_pose_to_sim(soup_pose)
        soup.write_root_velocity_to_sim(torch.zeros(1, 6, device=device))

        # Basket
        basket_pos_w = env_origin + torch.tensor(basket_rel, dtype=torch.float32, device=device)
        basket_quat_t = torch.tensor(basket_quat_wxyz, dtype=torch.float32, device=device)
        basket_pose = torch.cat([basket_pos_w, basket_quat_t]).unsqueeze(0)  # (1, 7)
        basket_obj.write_root_pose_to_sim(basket_pose)
        basket_obj.write_root_velocity_to_sim(torch.zeros(1, 6, device=device))

        # Propagate written state + stabilize
        for _ in range(args.stabilize_steps):
            env_unwrapped.scene.write_data_to_sim()
            env_unwrapped.sim.step(render=False)
            env_unwrapped.scene.update(env_unwrapped.physics_dt)
            ee_frame.update(env_unwrapped.physics_dt)

        # Capture initial scene state (relative) for HDF5
        initial_state = env_unwrapped.scene.get_state(is_relative=True)
        # Extract env 0 only
        initial_state_ep = _extract_env0(initial_state)

        # Read initial EEF pose (world frame, end_effector frame index 0)
        prev_ee_pos = ee_frame.data.target_pos_w[0, 0].clone()
        prev_ee_quat = ee_frame.data.target_quat_w[0, 0].clone()

        recorded_actions = []

        for step_idx, action_libero in enumerate(actions_libero):
            target_joints = libero_dof_pos_to_tensor(action_libero["dof_pos_target"], device)

            # Apply arm joint position targets (PD drives toward target over decimation steps)
            robot.set_joint_position_target(
                target_joints[:ARM_DOF].unsqueeze(0),
                joint_ids=arm_joint_ids,
            )
            robot.set_joint_position_target(
                target_joints[ARM_DOF:].unsqueeze(0),
                joint_ids=finger_joint_ids,
            )

            # Step physics (decimation=5 matches LIBERO's ~20Hz control)
            for _ in range(env_unwrapped.cfg.decimation):
                env_unwrapped.scene.write_data_to_sim()
                env_unwrapped.sim.step(render=False)
                env_unwrapped.scene.update(env_unwrapped.physics_dt)

            ee_frame.update(env_unwrapped.physics_dt)

            curr_ee_pos = ee_frame.data.target_pos_w[0, 0].clone()
            curr_ee_quat = ee_frame.data.target_quat_w[0, 0].clone()

            finger_pos = robot.data.joint_pos[0, finger_joint_ids]

            action = compute_delta_eef_action(
                prev_ee_pos, prev_ee_quat,
                curr_ee_pos, curr_ee_quat,
                finger_pos,
                args.gripper_threshold,
            )
            recorded_actions.append(action.cpu().numpy())

            prev_ee_pos = curr_ee_pos
            prev_ee_quat = curr_ee_quat

        # Add settling steps: let soup fall into basket after gripper opens.
        # Record as no-op actions (zero EEF delta, gripper open) so annotate_demos
        # replay includes the time needed for the soup to land.
        SETTLE_STEPS = 15
        noop_action = np.zeros(7, dtype=np.float32)
        noop_action[6] = 1.0  # gripper open
        for _ in range(SETTLE_STEPS):
            env_unwrapped.scene.write_data_to_sim()
            env_unwrapped.sim.step(render=False)
            env_unwrapped.scene.update(env_unwrapped.physics_dt)
            recorded_actions.append(noop_action.copy())

        # Check success after full episode + settling
        soup_pos = soup.data.root_pos_w[0]
        basket_pos_curr = basket_obj.data.root_pos_w[0]
        xy_dist = torch.linalg.norm(soup_pos[:2] - basket_pos_curr[:2])
        soup_z_local = soup_pos[2] - env_unwrapped.scene.env_origins[0, 2]
        basket_z_local = basket_pos_curr[2] - env_unwrapped.scene.env_origins[0, 2]
        in_z = (soup_z_local > basket_z_local + 0.01) and (soup_z_local < basket_z_local + 0.25)
        success = (xy_dist < 0.08) and in_z

        if success:
            demos.append({
                "initial_state": initial_state_ep,
                "actions": np.stack(recorded_actions),  # (T, 7)
            })
            print(f"  Recorded demo {len(demos)} ({len(recorded_actions)} steps)")
        else:
            print(f"  Episode failed (soup xy_dist={xy_dist:.3f} z_local={soup_z_local:.3f} basket_z={basket_z_local:.3f}), skipping")

    env.close()

    if len(demos) == 0:
        print("ERROR: No successful demos. Check LIBERO pkl path and object positions.")
        return

    write_hdf5(args.output_file, "Isaac-Franka-PickBasket-IK-Rel-Mimic-v0", demos)
    print(f"\nDone. {len(demos)} demos saved.")
    print(f"\nNext step — annotate:")
    print(
        f"  python IsaacLab/scripts/imitation_learning/isaaclab_mimic/annotate_demos.py"
        f" --input_file {args.output_file}"
        f" --output_file /tmp/pick_basket_src_annotated.hdf5"
        f" --env_id Isaac-Franka-PickBasket-IK-Rel-Mimic-v0"
        f" --auto --headless"
    )


def _extract_env0(state: dict) -> dict:
    """Recursively extract env index 0 from all tensors in a scene state dict."""
    result = {}
    for key, val in state.items():
        if isinstance(val, dict):
            result[key] = _extract_env0(val)
        elif isinstance(val, torch.Tensor):
            result[key] = val[[0]]  # keep shape (1, ...) for reset_to compatibility
        else:
            result[key] = val
    return result


if __name__ == "__main__":
    main()
    simulation_app.close()
