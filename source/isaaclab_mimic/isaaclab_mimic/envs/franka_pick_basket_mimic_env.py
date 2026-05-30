# Copyright (c) 2024-2026, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""MimicEnv wrapper for Franka pick-and-place-in-basket (IK relative control)."""

from collections.abc import Sequence

import torch

from .pick_place_mimic_env import PickPlaceRelMimicEnv


class FrankaPickBasketMimicEnv(PickPlaceRelMimicEnv):
    """
    Isaac Lab Mimic environment wrapper for the Franka pick-basket task.

    Two subtasks:
        1. grasp  — relative to alphabet_soup object frame
        2. lift   — relative to basket frame (robot carries object over basket)

    Subtask termination signals are read from ``obs_buf["subtask_terms"]``
    which is populated by the SubtaskCfg observation group defined in
    ``FrankaPickBasketMimicEnvCfg``.
    """

    def get_subtask_term_signals(self, env_ids: Sequence[int] | None = None) -> dict[str, torch.Tensor]:
        """Return grasp and lift subtask completion flags."""
        if env_ids is None:
            env_ids = slice(None)

        subtask_terms = self.obs_buf["subtask_terms"]
        return {
            "grasp": subtask_terms["grasp"][env_ids],
            "lift": subtask_terms["lift"][env_ids],
        }
