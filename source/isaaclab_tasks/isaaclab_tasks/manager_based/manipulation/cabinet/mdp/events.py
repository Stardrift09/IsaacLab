# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Custom reset events for the cabinet manipulation task."""

from __future__ import annotations

import torch

from isaaclab.assets import Articulation
from isaaclab.envs import ManagerBasedEnv
from isaaclab.managers import SceneEntityCfg


def reset_robot_joints_ik(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    # Dangerous zone in env-local coordinates (cabinet body region).
    # EEF landing inside this box is rejected and resampled.
    # Cabinet sits at local x=0.8; drawer face is ~x=0.50.
    danger_box_min: tuple[float, float, float] = (0.45, -0.25, 0.25),
    danger_box_max: tuple[float, float, float] = (0.90,  0.25, 0.85),
    # How far each arm joint may deviate from the default pose.
    position_range: float = 0.6,
    # Upper bound on rejection-sampling retries per env.
    max_resample_iters: int = 10,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> None:
    """Reset robot joints to a random valid pose with EEF outside the cabinet danger zone.

    Approach (following test_pick_it_up.py pattern)
    ------------------------------------------------
    1. Sample random arm joint positions (default ± ``position_range`` rad), clamped
       to soft joint limits.
    2. Write the joint state to sim, then call ``scene.update()`` to propagate FK
       so that ``robot.data.body_pos_w`` reflects the sampled configuration.
    3. Convert the EEF world position back to env-local coordinates and test whether
       it falls inside the cabinet danger box.
    4. For any env whose EEF landed inside the danger zone, resample and repeat up
       to ``max_resample_iters`` times.  Remaining violations (very rare) are left
       as-is — they are still kinematically valid, just sub-optimal for diversity.
    """
    robot: Articulation = env.scene[asset_cfg.name]
    device = robot.device
    n = len(env_ids)

    if n == 0:
        return

    # ---- indices ----------------------------------------------------------------
    arm_ids, _ = robot.find_joints("panda_joint.*")   # [0..6]
    body_ids, _ = robot.find_bodies("panda_hand")
    hand_idx = body_ids[0]

    # ---- joint limits -----------------------------------------------------------
    lim = robot.data.soft_joint_pos_limits[env_ids][:, arm_ids, :]  # (n, 7, 2)
    lim_lo = lim[..., 0]
    lim_hi = lim[..., 1]

    # ---- helper: sample one batch of arm joint positions ------------------------
    def _sample_arm_joints(ids_subset, lo, hi):
        n_sub = len(ids_subset)
        default = robot.data.default_joint_pos[ids_subset][:, arm_ids].clone()
        offset  = position_range * (2.0 * torch.rand(n_sub, len(arm_ids), device=device) - 1.0)
        return (default + offset).clamp(lo, hi)

    # ---- initial sample ---------------------------------------------------------
    arm_joint_pos = _sample_arm_joints(env_ids, lim_lo, lim_hi)  # (n, 7)

    all_joint_pos = robot.data.default_joint_pos[env_ids].clone()  # (n, 9)
    all_joint_pos[:, arm_ids] = arm_joint_pos
    zero_vel = torch.zeros_like(robot.data.default_joint_vel[env_ids])

    robot.write_joint_state_to_sim(all_joint_pos, zero_vel, env_ids=env_ids)
    env.scene.update(env.physics_dt)

    # ---- rejection-sampling loop ------------------------------------------------
    d_min = torch.tensor(danger_box_min, device=device)
    d_max = torch.tensor(danger_box_max, device=device)
    env_origins = env.scene.env_origins  # (N_total, 3)

    active_ids = env_ids  # indices into the full env array (not 0..n-1)
    active_lo  = lim_lo
    active_hi  = lim_hi

    for _ in range(max_resample_iters):
        ee_pos_w     = robot.data.body_pos_w[active_ids, hand_idx]            # (m, 3)
        ee_pos_local = ee_pos_w - env_origins[active_ids]                     # (m, 3)

        in_danger = (
            (ee_pos_local[:, 0] > d_min[0]) & (ee_pos_local[:, 0] < d_max[0]) &
            (ee_pos_local[:, 1] > d_min[1]) & (ee_pos_local[:, 1] < d_max[1]) &
            (ee_pos_local[:, 2] > d_min[2]) & (ee_pos_local[:, 2] < d_max[2])
        )

        if not in_danger.any():
            break

        bad_mask  = in_danger
        bad_ids   = active_ids[bad_mask]
        bad_lo    = active_lo[bad_mask]
        bad_hi    = active_hi[bad_mask]

        new_arm   = _sample_arm_joints(bad_ids, bad_lo, bad_hi)
        new_all   = robot.data.default_joint_pos[bad_ids].clone()
        new_all[:, arm_ids] = new_arm
        new_vel   = torch.zeros_like(robot.data.default_joint_vel[bad_ids])

        robot.write_joint_state_to_sim(new_all, new_vel, env_ids=bad_ids)
        env.scene.update(env.physics_dt)

        active_ids = bad_ids
        active_lo  = bad_lo
        active_hi  = bad_hi
