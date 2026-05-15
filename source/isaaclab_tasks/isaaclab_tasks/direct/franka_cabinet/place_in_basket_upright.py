# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Stage 2 (strict): Place the grasped object inside the basket while keeping rotation ≤ 10°."""

from __future__ import annotations

import math
import torch

from isaaclab.utils import configclass
from isaaclab.utils.math import quat_conjugate, quat_mul

from .place_in_basket import PlaceInBasket, PlaceInBasketCfg


@configclass
class PlaceInBasketUprightCfg(PlaceInBasketCfg):
    pass


class PlaceInBasketUpright(PlaceInBasket):
    """Placement stage: succeed only when inside basket AND object rotation ≤ 10° from default."""

    _UPRIGHT_THRESHOLD = 10.0 * math.pi / 180.0  # radians

    def _get_dones(self):
        terminated, truncated = super()._get_dones()

        current_rot = self.target_object.data.root_quat_w
        desired_rot = self.target_object.data.default_root_state[:, 3:7]
        current_rot_inv = quat_conjugate(current_rot)
        q_error = quat_mul(desired_rot, current_rot_inv)
        q_error = q_error / torch.norm(q_error, dim=-1, keepdim=True).clamp_min(1e-9)
        q_error = torch.where(q_error[:, 0:1] < 0, -q_error, q_error)
        angle_error = 2.0 * torch.acos(torch.clamp(q_error[:, 0], -1.0, 1.0))
        upright = angle_error < self._UPRIGHT_THRESHOLD

        return terminated & upright, truncated
