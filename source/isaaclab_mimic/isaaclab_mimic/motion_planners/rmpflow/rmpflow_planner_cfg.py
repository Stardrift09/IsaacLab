# Copyright (c) 2024-2026, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass, field

from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR


@dataclass
class RMPFlowPlannerCfg:
    """Configuration for the RMPFlow-based motion planner.

    RMPFlow is a reactive motion policy that drives the robot EEF toward a Cartesian
    target while avoiding obstacles in real time.  For offline pre-planning (required by
    MotionPlannerBase) the planner saves physics state, runs the controller for up to
    ``max_steps`` steps in a shadow pass, records the resulting EEF poses, then restores
    the physics state.  The recorded poses become the waypoints for the data generator.
    """

    # LULA robot description files for Franka Panda.
    # Override for other robots.
    rmpflow_config_file: str = (
        f"{ISAAC_NUCLEUS_DIR}/Robots/FrankaEmika/rmpflow/franka_rmpflow_common.yaml"
    )
    urdf_file: str = (
        f"{ISAAC_NUCLEUS_DIR}/Robots/FrankaEmika/lula_franka_gen.urdf"
    )
    collision_file: str = (
        f"{ISAAC_NUCLEUS_DIR}/Robots/FrankaEmika/rmpflow/robot_descriptor.yaml"
    )
    end_effector_frame_name: str = "panda_hand"

    # Controller parameters
    evaluations_per_frame: float = 5.0
    ignore_robot_state_updates: bool = False

    # Planning parameters
    max_steps: int = 200
    goal_tolerance_m: float = 0.02
    goal_rot_tolerance_rad: float = 0.1

    # Which robot prim path expression to use inside the env
    # Use {ENV_REGEX_NS} pattern: replaced at runtime with each env's prim path.
    prim_paths_expr: str = "/World/envs/env_.*/Robot"
