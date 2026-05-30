# Copyright (c) 2024-2026, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""RMPFlow-based motion planner implementing MotionPlannerBase.

Uses LULA's RmpFlow to reactively plan toward a Cartesian target.  Because
MotionPlannerBase requires pre-planned waypoints (offline), the planner performs
a *shadow pass*:

  1. Save the full environment physics state.
  2. Run RmpFlowController for up to ``max_steps`` steps toward the target.
  3. At each step record the current EEF pose.
  4. Stop early if the EEF is within ``goal_tolerance_m`` of the target.
  5. Restore the original physics state.
  6. Return recorded poses as waypoints.

This produces reactive, collision-avoiding interpolation paths that are different
from CuRobo's optimal trajectories, increasing diversity in the generated dataset.
"""

from __future__ import annotations

import logging
from typing import Any

import torch

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation
from isaaclab.controllers.rmp_flow import RmpFlowController, RmpFlowControllerCfg
from isaaclab.envs.manager_based_env import ManagerBasedEnv
from isaaclab.sensors import FrameTransformer

from ..motion_planner_base import MotionPlannerBase
from .rmpflow_planner_cfg import RMPFlowPlannerCfg

logger = logging.getLogger(__name__)


class RMPFlowPlanner(MotionPlannerBase):
    """Reactive motion planner using LULA's RmpFlow.

    Produces diverse interpolation paths compared to CuRobo (which optimises for
    shortest collision-free path).  RmpFlow's reactive nature creates slightly
    curved, dynamically natural trajectories.

    Args:
        env: The IsaacLab ManagerBasedEnv instance.
        robot: The robot Articulation asset.
        cfg: RMPFlowPlannerCfg specifying files and parameters.
        env_id: Index of the parallel environment to plan for (default 0).
        debug: Enable verbose logging.
    """

    def __init__(
        self,
        env: ManagerBasedEnv,
        robot: Articulation,
        cfg: RMPFlowPlannerCfg,
        env_id: int = 0,
        debug: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(env=env, robot=robot, env_id=env_id, debug=debug)
        self.cfg = cfg

        # Build the underlying RmpFlowController (same one used by RMPFlowAction)
        controller_cfg = RmpFlowControllerCfg(
            name="rmp_flow",
            config_file=cfg.rmpflow_config_file,
            urdf_file=cfg.urdf_file,
            collision_file=cfg.collision_file,
            frame_name=cfg.end_effector_frame_name,
            evaluations_per_frame=cfg.evaluations_per_frame,
            ignore_robot_state_updates=cfg.ignore_robot_state_updates,
        )
        self._controller = RmpFlowController(cfg=controller_cfg, device=env.device)
        self._controller.initialize(cfg.prim_paths_expr)

        # Locate the EEF frame sensor in the env (used to read EEF poses)
        self._ee_frame: FrameTransformer | None = None
        for key in env.scene.keys():
            asset = env.scene[key]
            if isinstance(asset, FrameTransformer):
                self._ee_frame = asset
                break
        if self._ee_frame is None:
            raise RuntimeError(
                "RMPFlowPlanner: No FrameTransformer found in the scene. "
                "Ensure the env cfg includes an ee_frame sensor."
            )

        self._waypoints: list[torch.Tensor] = []
        self._idx: int = 0

    # ------------------------------------------------------------------
    # MotionPlannerBase interface
    # ------------------------------------------------------------------

    def update_world_and_plan_motion(self, target_pose: torch.Tensor, **kwargs: Any) -> bool:
        """Shadow-pass: run RMPFlow to target, capture EEF poses, restore state.

        Args:
            target_pose: (4, 4) homogeneous target EEF pose for env ``env_id``.

        Returns:
            True if at least one waypoint was recorded (planning succeeded).
        """
        self.reset_plan()

        target_pos, target_rot = math_utils.unmake_pose(target_pose.unsqueeze(0))
        target_pos_np = target_pos[0].cpu().numpy()
        target_quat = math_utils.quat_from_matrix(target_rot)[0]  # w,x,y,z
        # RmpFlow expects pos + quat(w,x,y,z) → pack as 7-vec command
        target_quat_np = target_quat.cpu().numpy()

        # --- 1. Save physics state ---
        saved_state = self.env.scene.get_state(is_relative=False)

        # --- 2. Shadow pass ---
        command = torch.zeros(self._controller.num_robots, 7, device=self.env.device)
        command[self.env_id, 0:3] = target_pos[0]
        command[self.env_id, 3:7] = target_quat

        self._controller.set_command(command)

        for _ in range(self.cfg.max_steps):
            # Compute next joint targets from RMPFlow
            dof_pos_target, dof_vel_target = self._controller.compute()

            # Apply only for the env_id robot, write directly to articulation
            self.robot.set_joint_position_target(
                dof_pos_target[[self.env_id]],
                env_ids=torch.tensor([self.env_id], device=self.env.device),
            )

            # Step physics (all envs but only env_id matters for recording)
            self.env.sim.step(render=False)
            self.env.scene.update(dt=self.env.physics_dt)
            self._ee_frame.update(dt=self.env.physics_dt)

            # Record EEF pose for env_id
            ee_pos_w = self._ee_frame.data.target_pos_w[self.env_id, 0, :]
            ee_quat_w = self._ee_frame.data.target_quat_w[self.env_id, 0, :]
            ee_pose = math_utils.make_pose(ee_pos_w.unsqueeze(0), math_utils.matrix_from_quat(ee_quat_w.unsqueeze(0)))[0]
            self._waypoints.append(ee_pose.clone())

            # Check convergence
            dist = torch.linalg.norm(ee_pos_w - target_pos[0].to(self.env.device))
            if dist < self.cfg.goal_tolerance_m:
                if self.debug:
                    logger.debug(f"RMPFlowPlanner: converged in {len(self._waypoints)} steps")
                break

        # --- 3. Restore physics state ---
        self.env.scene.set_state(saved_state, is_relative=False)
        # Re-sync scene after state restore
        self.env.sim.step(render=False)
        self.env.scene.update(dt=self.env.physics_dt)

        if self.debug and len(self._waypoints) == 0:
            logger.warning("RMPFlowPlanner: no waypoints generated")

        return len(self._waypoints) > 0

    def has_next_waypoint(self) -> bool:
        return self._idx < len(self._waypoints)

    def get_next_waypoint_ee_pose(self) -> torch.Tensor:
        pose = self._waypoints[self._idx]
        self._idx += 1
        return pose

    def reset_plan(self) -> None:
        self._waypoints = []
        self._idx = 0
        self._controller.reset_idx()

    def get_planner_info(self) -> dict[str, Any]:
        info = super().get_planner_info()
        info.update(
            {
                "type": "RMPFlow",
                "max_steps": self.cfg.max_steps,
                "goal_tolerance_m": self.cfg.goal_tolerance_m,
                "waypoints_recorded": len(self._waypoints),
            }
        )
        return info
