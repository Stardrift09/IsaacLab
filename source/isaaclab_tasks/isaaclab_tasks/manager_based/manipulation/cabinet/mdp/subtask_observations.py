# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.sensors import FrameTransformerData

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def drawer_opened(env: ManagerBasedRLEnv, threshold: float = 0.39) -> torch.Tensor:
    """True when the top drawer joint position exceeds threshold metres (success termination)."""
    cabinet = env.scene["cabinet"]
    joint_idx = cabinet.find_joints("drawer_top_joint")[0][0]
    return cabinet.data.joint_pos[:, joint_idx] > threshold


def drawer_approached(env: ManagerBasedRLEnv, threshold: float = 0.1) -> torch.Tensor:
    """True when the EEF is within threshold metres of the drawer handle grasp point.

    Reads the ``ee_frame`` and ``cabinet_frame`` FrameTransformers which must be
    present in the scene (they are in CabinetSceneCfg by default).
    """
    ee_tf_data: FrameTransformerData = env.scene["ee_frame"].data
    cabinet_tf_data: FrameTransformerData = env.scene["cabinet_frame"].data

    dist = torch.linalg.vector_norm(
        cabinet_tf_data.target_pos_w[..., 0, :] - ee_tf_data.target_pos_w[..., 0, :],
        dim=-1,
    )
    return dist < threshold
