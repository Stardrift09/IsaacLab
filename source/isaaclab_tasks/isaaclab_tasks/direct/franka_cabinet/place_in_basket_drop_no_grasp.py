"""PlaceInBasketDrop ablation: no grasp/stage detection utilities.

Terminates when object is inside basket AND EEF is currently outside the exclusion zone.
No release detection heuristics — EEF position at termination is the only constraint.
"""

from __future__ import annotations

import torch

from isaaclab.utils import configclass

from .place_in_basket_no_grasp import PlaceInBasketNoGrasp, PlaceInBasketNoGraspCfg


@configclass
class PlaceInBasketDropNoGraspCfg(PlaceInBasketNoGraspCfg):
    # EEF exclusion radius = factor * target_site_radius
    drop_eef_exclusion_factor: float = 2.0


class PlaceInBasketDropNoGrasp(PlaceInBasketNoGrasp):
    """PlaceInBasketDrop ablation: succeed when object inside basket and EEF outside exclusion zone."""

    def _get_dones(self):
        terminated, truncated = super()._get_dones()

        eef_xy = self.robot_grasp_pos[:, :2]
        basket_xy = self.target_site.data.root_pos_w[:, :2]
        exclusion_radius = self.cfg.drop_eef_exclusion_factor * self.target_site_radius
        eef_outside = ((eef_xy - basket_xy) ** 2).sum(-1) >= exclusion_radius ** 2

        return terminated & eef_outside, truncated
