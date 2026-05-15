# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Stage 2 (drop variant): Start grasped, drop object while EEF is outside basket XY, land inside."""

from __future__ import annotations

import torch

from isaaclab.utils import configclass

from .place_in_basket import PlaceInBasket, PlaceInBasketCfg


@configclass
class PlaceInBasketDropCfg(PlaceInBasketCfg):
    # Multiplier on target_site_radius for the EEF exclusion zone.
    # target_site_radius is object-clearance-based and very small; a larger zone
    # forces the robot to genuinely release from outside the basket area.
    drop_eef_exclusion_factor: float = 2.0


class PlaceInBasketDrop(PlaceInBasket):
    """Placement stage: succeed only when object dropped outside basket XY footprint and lands inside."""

    def __init__(self, cfg: PlaceInBasketDropCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self.prev_grasped = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self.drop_outside_triggered = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)

    def _get_dones(self):
        terminated, truncated = super()._get_dones()

        eef_xy = self.robot_grasp_pos[:, :2]
        basket_xy = self.target_site.data.root_pos_w[:, :2]
        exclusion_radius = self.cfg.drop_eef_exclusion_factor * self.target_site_radius
        eef_over_basket = ((eef_xy - basket_xy) ** 2).sum(-1) < exclusion_radius ** 2

        drop_event = self.prev_grasped & ~self.grasped & ~eef_over_basket
        self.drop_outside_triggered |= drop_event
        self.prev_grasped = self.grasped.clone()

        return terminated & self.drop_outside_triggered, truncated

    def _reset_idx(self, env_ids: torch.Tensor | None):
        super()._reset_idx(env_ids)
        self.prev_grasped[env_ids] = False
        self.drop_outside_triggered[env_ids] = False
