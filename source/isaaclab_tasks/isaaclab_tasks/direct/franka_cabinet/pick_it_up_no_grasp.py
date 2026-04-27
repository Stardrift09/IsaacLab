"""Ablation: PickItUp without grasp-detection and stage-detection utilities.

Removes _grasp_detection, _current_stage_detection, and _pregrasp_detection.
LLM must shape reward from raw observations only.
Terminates when object is lifted above basket rim with small rotation (no grasp check).
"""

from __future__ import annotations

import torch

from isaaclab.utils import configclass
from .pick_it_up import PickItUp, PickItUpCfg
from .test_pick_it_up import TestPickItUp


@configclass
class PickItUpNoGraspCfg(PickItUpCfg):
    pass


class PickItUpNoGrasp(PickItUp):
    """Ablation variant: no _grasp_detection, no _current_stage_detection."""

    def _grasp_detection(self, env_ids=None):
        if env_ids is None:
            env_ids = self._robot._ALL_INDICES
        return torch.zeros(len(env_ids), device=self.device, dtype=torch.bool)

    def _pregrasp_detection(self, env_ids=None):
        if env_ids is None:
            env_ids = self._robot._ALL_INDICES
        return torch.zeros(len(env_ids), device=self.device, dtype=torch.bool)

    def _current_stage_detection(self, env_ids=None):
        if env_ids is None:
            env_ids = self._robot._ALL_INDICES
        return torch.zeros((len(env_ids), self.num_stages), device=self.device)

    def _get_dones(self):
        # Run TestPickItUp logic directly to populate intermediate values
        # (high_enough, small_rotation, etc.) without PickItUp's grasped requirement.
        _, truncated = TestPickItUp._get_dones(self)
        terminated = self.high_enough & self.small_rotation
        print(self.small_rotation)
        return terminated, truncated

    def _get_observations(self) -> dict:
        """
        All texts in _get_observations() are very important hints for the task!

        Objects' root_pos_w is their center, and site's root_pos_w is the bottom center.
        self.rigid_objects is a dict with all RigidObjects.
        Action space is 8-dim; last dim > 0 means opening gripper.
        self.to_desired_rot: quaternion from current to desired grasp orientation.
            High reward when first element (w) approaches 1.
        self.helper_variable = torch.zeros((self.num_envs, 10), device=self.device)
        self.target_site_corners_world: [2, 3] tensor, first row = min corner, second = max corner (env-local frame).

        NOTE: _grasp_detection(), _current_stage_detection(), and self.stage are NOT available.
        Shape rewards using raw geometry: finger positions, object height, distances, contact forces via
        self.scene['left_contact_sensor'] and self.scene['right_contact_sensor'].
        """
        dof_pos_scaled = (
            2.0
            * (self._robot.data.joint_pos - self.robot_dof_lower_limits)
            / (self.robot_dof_upper_limits - self.robot_dof_lower_limits)
            - 1.0
        )
        self.corners_target_obj_to_hand_pos = (
            self.corners_target_obj - self.robot_grasp_pos.unsqueeze(1)
        ).reshape(self.num_envs, -1)
        self.target_to_hand_pos = (
            self.target_object.data.root_pos_w - self.robot_grasp_pos
        )
        tcp_vel = (
            self._robot.data.body_link_lin_vel_w[:, self.left_finger_body_idx]
            + self._robot.data.body_link_lin_vel_w[:, self.right_finger_body_idx]
        ) / 2
        target_to_hand_vel = self.target_object.data.root_lin_vel_w - tcp_vel
        self.site_to_target_pos = (
            self.target_site.data.root_pos_w - self.target_object.data.root_pos_w
        )
        site_to_target_vel = (
            self.target_object.data.root_lin_vel_w - self.target_site.data.root_lin_vel_w
        )
        obs = torch.cat(
            (
                dof_pos_scaled,
                self._robot.data.joint_vel * self.cfg.dof_velocity_scale,
                self.corners_target_obj_to_hand_pos,
                self.target_to_hand_pos,
                self.to_desired_rot,
                self.site_to_target_pos,
                target_to_hand_vel,
                site_to_target_vel,
                self.prev_actions - self.actions,
            ),
            dim=-1,
        )
        return {"policy": torch.clamp(obs, -5.0, 5.0)}
