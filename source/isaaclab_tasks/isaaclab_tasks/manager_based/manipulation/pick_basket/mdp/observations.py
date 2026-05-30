# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformer

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def object_pos_in_world(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("alphabet_soup"),
) -> torch.Tensor:
    obj: RigidObject = env.scene[object_cfg.name]
    return obj.data.root_pos_w - env.scene.env_origins


def basket_pos_in_world(
    env: ManagerBasedRLEnv,
    basket_cfg: SceneEntityCfg = SceneEntityCfg("basket"),
) -> torch.Tensor:
    basket: RigidObject = env.scene[basket_cfg.name]
    return basket.data.root_pos_w - env.scene.env_origins


def object_grasped_and_lifted(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg,
    ee_frame_cfg: SceneEntityCfg,
    object_cfg: SceneEntityCfg,
    lift_height_threshold: float = 0.12,
) -> torch.Tensor:
    """True when the object is grasped AND lifted above lift_height_threshold (env-local z)."""
    robot: Articulation = env.scene[robot_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    obj: RigidObject = env.scene[object_cfg.name]

    object_pos_w = obj.data.root_pos_w
    ee_pos_w = ee_frame.data.target_pos_w[:, 0, :]
    pose_diff = torch.linalg.vector_norm(object_pos_w - ee_pos_w, dim=1)

    gripper_joint_ids, _ = robot.find_joints(env.cfg.gripper_joint_names)
    grasped = torch.logical_and(
        pose_diff < 0.08,
        torch.abs(
            robot.data.joint_pos[:, gripper_joint_ids[0]]
            - torch.tensor(env.cfg.gripper_open_val, dtype=torch.float32, device=env.device)
        ) > env.cfg.gripper_threshold,
    )
    grasped = torch.logical_and(
        grasped,
        torch.abs(
            robot.data.joint_pos[:, gripper_joint_ids[1]]
            - torch.tensor(env.cfg.gripper_open_val, dtype=torch.float32, device=env.device)
        ) > env.cfg.gripper_threshold,
    )

    object_z_local = object_pos_w[:, 2] - env.scene.env_origins[:, 2]
    lifted = object_z_local > lift_height_threshold
    return torch.logical_and(grasped, lifted)


def object_in_basket(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("alphabet_soup"),
    basket_cfg: SceneEntityCfg = SceneEntityCfg("basket"),
    xy_radius: float = 0.08,
    min_z_above_basket_bottom: float = 0.01,
    max_z_above_basket_rim: float = 0.15,
    basket_height: float = 0.10,
) -> torch.Tensor:
    """True when object XY is within xy_radius of basket center AND at appropriate height."""
    obj: RigidObject = env.scene[object_cfg.name]
    basket: RigidObject = env.scene[basket_cfg.name]

    obj_pos_w = obj.data.root_pos_w
    basket_pos_w = basket.data.root_pos_w

    xy_dist = torch.linalg.vector_norm(obj_pos_w[:, :2] - basket_pos_w[:, :2], dim=1)
    obj_z_local = obj_pos_w[:, 2] - env.scene.env_origins[:, 2]
    basket_z_local = basket_pos_w[:, 2] - env.scene.env_origins[:, 2]

    in_xy = xy_dist < xy_radius
    in_z = torch.logical_and(
        obj_z_local > basket_z_local + min_z_above_basket_bottom,
        obj_z_local < basket_z_local + basket_height + max_z_above_basket_rim,
    )
    return torch.logical_and(in_xy, in_z)
