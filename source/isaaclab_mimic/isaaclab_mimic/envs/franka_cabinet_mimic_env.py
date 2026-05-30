# Copyright (c) 2024-2026, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""MimicEnv wrapper for Franka cabinet open-drawer (IK relative control)."""

from collections.abc import Sequence

import torch

from .pick_place_mimic_env import PickPlaceRelMimicEnv


class FrankaCabinetMimicEnv(PickPlaceRelMimicEnv):
    """Isaac Lab Mimic environment wrapper for the Franka open-drawer task.

    One subtask boundary:
        approached — EEF within 0.1 m of drawer handle.

    Segment 1 (approach): motion planner brings the gripper to the handle;
    waypoints warped relative to the cabinet root pose for new cabinet positions.

    Segment 2 (open): actions replayed from the source demo and warped by the
    delta in cabinet root pose — the policy the user trains takes over here.
    """

    def get_subtask_term_signals(self, env_ids: Sequence[int] | None = None) -> dict[str, torch.Tensor]:
        if env_ids is None:
            env_ids = slice(None)
        subtask_terms = self.obs_buf["subtask_terms"]
        return {"approached": subtask_terms["approached"][env_ids]}
