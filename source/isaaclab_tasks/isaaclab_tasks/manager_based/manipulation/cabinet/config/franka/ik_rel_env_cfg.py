# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.envs.mdp.actions.actions_cfg import DifferentialInverseKinematicsActionCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

from isaaclab_tasks.manager_based.manipulation.cabinet import mdp

from . import joint_pos_env_cfg

##
# Pre-defined configs
##
from isaaclab_assets.robots.franka import FRANKA_PANDA_HIGH_PD_CFG  # isort: skip


@configclass
class FrankaCabinetEnvCfg(joint_pos_env_cfg.FrankaCabinetEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # Set Franka as robot
        # We switch here to a stiffer PD controller for IK tracking to be better.
        self.scene.robot = FRANKA_PANDA_HIGH_PD_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # Set actions for the specific robot type (franka)
        self.actions.arm_action = DifferentialInverseKinematicsActionCfg(
            asset_name="robot",
            joint_names=["panda_joint.*"],
            body_name="panda_hand",
            controller=DifferentialIKControllerCfg(command_type="pose", use_relative_mode=True, ik_method="dls"),
            scale=0.5,
            body_offset=DifferentialInverseKinematicsActionCfg.OffsetCfg(pos=[0.0, 0.0, 0.107]),
        )

        # Replace default rewards with Eureka-discovered reward (~90% success)
        self.rewards.approach_ee_handle = None
        self.rewards.align_ee_handle = None
        self.rewards.approach_gripper_handle = None
        self.rewards.align_grasp_around_handle = None
        self.rewards.grasp_handle = None
        self.rewards.open_drawer_bonus = None
        self.rewards.multi_stage_open_drawer = None
        self.rewards.action_rate_l2 = None
        self.rewards.joint_vel = None
        self.rewards.eureka_open_drawer = RewTerm(
            func=mdp.eureka_open_drawer,
            weight=1.0,
            params={"asset_cfg": SceneEntityCfg("cabinet", joint_names=["drawer_top_joint"])},
        )

        # Terminate on success (drawer_top_joint > 0.39 m)
        self.terminations.success = DoneTerm(func=mdp.drawer_opened, params={"threshold": 0.39})

        # Replace narrow ±0.1 rad joint-offset reset with IK-based reset.
        # Samples a random EEF target in a safe workspace (x < 0.55 m, away from
        # the cabinet at x = 0.8 m) and solves IK to get diverse, valid initial
        # joint configurations.
        self.events.reset_robot_joints = EventTerm(
            func=mdp.reset_robot_joints_ik,
            mode="reset",
            params={
                "danger_box_min": (0.45, -0.25, 0.25),
                "danger_box_max": (0.90,  0.25, 0.85),
                "position_range": 0.6,
                "max_resample_iters": 10,
            },
        )

        # Cabinet position stays fixed; only yaw is randomised so the robot sees
        # the drawer handle at different angles each episode.
        self.events.reset_cabinet_pose = EventTerm(
            func=mdp.reset_root_state_uniform,
            mode="reset",
            params={
                "pose_range": {"yaw": (-0.26, 0.26)},  # ±15 degrees
                "velocity_range": {},
                "asset_cfg": SceneEntityCfg("cabinet"),
            },
        )


@configclass
class FrankaCabinetEnvCfg_PLAY(FrankaCabinetEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # disable randomization for play
        self.observations.policy.enable_corruption = False
