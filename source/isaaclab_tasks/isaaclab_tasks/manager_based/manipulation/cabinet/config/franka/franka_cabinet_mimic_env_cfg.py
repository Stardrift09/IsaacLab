# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Franka cabinet MimicGen env cfg: extends IK-rel env with subtask signals and cabinet randomisation."""

from isaaclab.envs.mimic_env_cfg import MimicEnvCfg, SubTaskConfig
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from isaaclab_tasks.manager_based.manipulation.cabinet import mdp

from . import ik_rel_env_cfg


# ---------------------------------------------------------------------------
# Observations override: non-concatenated policy + subtask signal group
# ---------------------------------------------------------------------------

@configclass
class _PolicyCfg(ObsGroup):
    """Same terms as CabinetEnvCfg.PolicyCfg but with concatenate_terms=False.

    eef_pos / eef_quat are required by FrankaCubeStackIKRelMimicEnv.get_robot_eef_pose
    which annotate_demos.py calls to record EEF waypoints.
    """

    joint_pos = ObsTerm(func=mdp.joint_pos_rel)
    joint_vel = ObsTerm(func=mdp.joint_vel_rel)
    cabinet_joint_pos = ObsTerm(
        func=mdp.joint_pos_rel,
        params={"asset_cfg": SceneEntityCfg("cabinet", joint_names=["drawer_top_joint"])},
    )
    cabinet_joint_vel = ObsTerm(
        func=mdp.joint_vel_rel,
        params={"asset_cfg": SceneEntityCfg("cabinet", joint_names=["drawer_top_joint"])},
    )
    rel_ee_drawer_distance = ObsTerm(func=mdp.rel_ee_drawer_distance)
    actions = ObsTerm(func=mdp.last_action)
    eef_pos = ObsTerm(func=mdp.ee_pos)
    eef_quat = ObsTerm(func=mdp.ee_quat)

    def __post_init__(self):
        self.enable_corruption = False
        self.concatenate_terms = False


@configclass
class _SubtaskCfg(ObsGroup):
    """Single subtask boundary: EEF within 0.1 m of drawer handle."""

    approached = ObsTerm(func=mdp.drawer_approached, params={"threshold": 0.1})

    def __post_init__(self):
        self.enable_corruption = False
        self.concatenate_terms = False


@configclass
class _ObservationsCfg:
    policy: _PolicyCfg = _PolicyCfg()
    subtask_terms: _SubtaskCfg = _SubtaskCfg()


# ---------------------------------------------------------------------------
# Mimic env cfg
# ---------------------------------------------------------------------------

@configclass
class FrankaCabinetMimicEnvCfg(ik_rel_env_cfg.FrankaCabinetEnvCfg, MimicEnvCfg):
    """Franka cabinet IK-rel MimicGen config.

    Subtask decomposition (1 boundary, 2 segments):
        1. approach  — EEF moves to drawer handle (object_ref=cabinet).
                       Motion planner handles this segment for new cabinet positions.
        2. open      — Policy / replayed actions pull the drawer open (object_ref=cabinet).
                       Warped by delta in cabinet root pose.
    """

    def __post_init__(self):
        super().__post_init__()

        # Replace observations with MimicGen-compatible version
        self.observations = _ObservationsCfg()

        # Cabinet pose randomisation is disabled here so that annotate_demos.py
        # can replay source demos at their recorded positions.
        # Re-enable for generate_dataset.py by setting this event in a subclass.
        # self.events.randomize_cabinet_pose = EventTerm(...)

        # ---- MimicGen data-gen settings ----
        self.datagen_config.name = "demo_src_cabinet_franka_ik_rel"
        self.datagen_config.generation_guarantee = True
        self.datagen_config.generation_keep_failed = True
        self.datagen_config.generation_num_trials = 10
        self.datagen_config.generation_select_src_per_subtask = True
        self.datagen_config.generation_transform_first_robot_pose = False
        self.datagen_config.generation_interpolate_from_last_target_pose = True
        self.datagen_config.generation_relative = True
        self.datagen_config.max_num_failures = 25
        self.datagen_config.seed = 1

        subtask_configs = []

        # Segment 1: approach to drawer handle.
        # Motion planner interpolates from start to the approach waypoint,
        # warped by delta in cabinet root pose.
        subtask_configs.append(
            SubTaskConfig(
                object_ref="cabinet",
                subtask_term_signal="approached",
                subtask_term_offset_range=(3, 10),
                selection_strategy="nearest_neighbor_object",
                selection_strategy_kwargs={"nn_k": 3},
                action_noise=0.03,
                num_interpolation_steps=5,
                num_fixed_steps=0,
                apply_noise_during_interpolation=False,
                description="Approach drawer handle",
                next_subtask_description="Pull drawer open",
            )
        )

        # Segment 2: pull drawer open.
        # Actions replayed from source demo, warped by cabinet pose delta.
        # subtask_term_signal=None because this is the final segment.
        subtask_configs.append(
            SubTaskConfig(
                object_ref="cabinet",
                subtask_term_signal=None,
                subtask_term_offset_range=(0, 0),
                selection_strategy="nearest_neighbor_object",
                selection_strategy_kwargs={"nn_k": 3},
                action_noise=0.02,
                num_interpolation_steps=5,
                num_fixed_steps=0,
                apply_noise_during_interpolation=False,
                description="Pull drawer open",
            )
        )

        self.subtask_configs["franka"] = subtask_configs
