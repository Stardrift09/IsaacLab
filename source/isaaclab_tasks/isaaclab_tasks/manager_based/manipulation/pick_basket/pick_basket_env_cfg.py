# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Manager-based pick-and-place-in-basket environment using LIBERO objects."""

import os
from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import FrameTransformerCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg, UsdFileCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg

from . import mdp

# Path to LIBERO HOPE object USD assets
_LIBERO_OBJECTS_ROOT = os.path.join(
    os.path.dirname(__file__), *[".."] * 7, "libero", "COMMON", "stable_hope_objects"
)
_ALPHABET_SOUP_USD = os.path.join(_LIBERO_OBJECTS_ROOT, "alphabet_soup", "usd", "alphabet_soup.usd")
_BASKET_USD = os.path.join(_LIBERO_OBJECTS_ROOT, "basket", "usd", "basket.usd")


##
# Scene definition
##

@configclass
class PickBasketSceneCfg(InteractiveSceneCfg):
    """Scene with Franka, an alphabet_soup object, and a basket."""

    robot: ArticulationCfg = MISSING
    ee_frame: FrameTransformerCfg = MISSING

    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0.5, 0, 0], rot=[0.707, 0, 0, 0.707]),
        spawn=UsdFileCfg(usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd"),
    )

    plane = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0, 0, -1.05]),
        spawn=GroundPlaneCfg(),
    )

    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )

    # Target object — randomised position set by events
    alphabet_soup = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/AlphabetSoup",
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=[0.40, 0.0, 0.02],
            # 90° around X so the can stands upright
            rot=[0.7071, 0.7071, 0.0, 0.0],
        ),
        spawn=UsdFileCfg(
            usd_path=_ALPHABET_SOUP_USD,
            rigid_props=RigidBodyPropertiesCfg(
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=1,
                max_angular_velocity=1000.0,
                max_linear_velocity=1000.0,
                max_depenetration_velocity=5.0,
                disable_gravity=False,
            ),
        ),
    )

    # Basket — fixed position (not randomised)
    basket = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Basket",
        init_state=RigidObjectCfg.InitialStateCfg(pos=[0.55, -0.22, 0.0], rot=[1, 0, 0, 0]),
        spawn=UsdFileCfg(
            usd_path=_BASKET_USD,
            rigid_props=RigidBodyPropertiesCfg(
                disable_gravity=False,
                max_depenetration_velocity=5.0,
            ),
        ),
    )


##
# MDP settings
##

@configclass
class ActionsCfg:
    arm_action: mdp.DifferentialInverseKinematicsActionCfg = MISSING
    gripper_action: mdp.BinaryJointPositionActionCfg = MISSING


@configclass
class ObservationsCfg:

    @configclass
    class PolicyCfg(ObsGroup):
        actions = ObsTerm(func=mdp.last_action)
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        eef_pos = ObsTerm(func=mdp.ee_frame_pos)
        eef_quat = ObsTerm(func=mdp.ee_frame_quat)
        gripper_pos = ObsTerm(func=mdp.gripper_pos)
        object_pos = ObsTerm(
            func=mdp.object_pos_in_world,
            params={"object_cfg": SceneEntityCfg("alphabet_soup")},
        )
        basket_pos = ObsTerm(
            func=mdp.basket_pos_in_world,
            params={"basket_cfg": SceneEntityCfg("basket")},
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    @configclass
    class SubtaskCfg(ObsGroup):
        """Subtask termination signals consumed by isaaclab_mimic."""

        grasp = ObsTerm(
            func=mdp.object_grasped,
            params={
                "robot_cfg": SceneEntityCfg("robot"),
                "ee_frame_cfg": SceneEntityCfg("ee_frame"),
                "object_cfg": SceneEntityCfg("alphabet_soup"),
            },
        )
        lift = ObsTerm(
            func=mdp.object_grasped_and_lifted,
            params={
                "robot_cfg": SceneEntityCfg("robot"),
                "ee_frame_cfg": SceneEntityCfg("ee_frame"),
                "object_cfg": SceneEntityCfg("alphabet_soup"),
                "lift_height_threshold": 0.12,
            },
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    policy: PolicyCfg = PolicyCfg()
    subtask_terms: SubtaskCfg = SubtaskCfg()


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    object_dropping = DoneTerm(
        func=mdp.root_height_below_minimum,
        params={"minimum_height": -0.05, "asset_cfg": SceneEntityCfg("alphabet_soup")},
    )

    success = DoneTerm(
        func=mdp.object_in_basket,
        params={
            "object_cfg": SceneEntityCfg("alphabet_soup"),
            "basket_cfg": SceneEntityCfg("basket"),
        },
    )


@configclass
class PickBasketEnvCfg(ManagerBasedRLEnvCfg):
    """Base configuration for pick-and-place-in-basket."""

    scene: PickBasketSceneCfg = PickBasketSceneCfg(num_envs=1024, env_spacing=2.5, replicate_physics=False)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    commands = None
    rewards = None
    events = None
    curriculum = None

    def __post_init__(self):
        self.decimation = 5
        self.episode_length_s = 20.0
        self.sim.dt = 0.01
        self.sim.render_interval = 2
        self.sim.physx.bounce_threshold_velocity = 0.2
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 1024 * 1024 * 4
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 16 * 1024
        self.sim.physx.friction_correlation_distance = 0.00625
