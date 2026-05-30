# Copyright (c) 2024-2026, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""MimicEnvCfg for Franka pick-and-place-in-basket (IK relative control)."""

from isaaclab.envs.mimic_env_cfg import MimicEnvCfg, SubTaskConfig
from isaaclab.utils import configclass

from isaaclab_tasks.manager_based.manipulation.pick_basket.config.franka.pick_basket_ik_rel_env_cfg import (
    FrankaPickBasketEnvCfg,
)


@configclass
class FrankaPickBasketMimicEnvCfg(FrankaPickBasketEnvCfg, MimicEnvCfg):
    """
    Isaac Lab Mimic config for the Franka pick-basket task.

    Subtask decomposition:
        1. grasp  — approach and grasp alphabet_soup.
                    Object reference frame: alphabet_soup.
        2. lift+place — carry object over basket and release.
                    Object reference frame: basket.
    """

    def __post_init__(self):
        super().__post_init__()

        self.datagen_config.name = "demo_src_pick_basket_franka_ik_rel"
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

        # Subtask 1: grasp the object
        # The source segment captures approach-to-grasp motion relative to the object.
        subtask_configs.append(
            SubTaskConfig(
                object_ref="alphabet_soup",
                subtask_term_signal="grasp",
                subtask_term_offset_range=(5, 15),
                selection_strategy="nearest_neighbor_object",
                selection_strategy_kwargs={"nn_k": 3},
                action_noise=0.03,
                num_interpolation_steps=5,
                num_fixed_steps=0,
                apply_noise_during_interpolation=False,
                description="Grasp the alphabet soup can",
                next_subtask_description="Carry object over basket and release",
            )
        )

        # Subtask 2: carry grasped object over basket and release
        # Relative to basket frame so new basket positions are handled.
        # subtask_term_signal=None because this is the final subtask.
        subtask_configs.append(
            SubTaskConfig(
                object_ref="basket",
                subtask_term_signal=None,
                subtask_term_offset_range=(0, 0),
                selection_strategy="nearest_neighbor_object",
                selection_strategy_kwargs={"nn_k": 3},
                action_noise=0.02,
                num_interpolation_steps=5,
                num_fixed_steps=0,
                apply_noise_during_interpolation=False,
                description="Place object in basket",
            )
        )

        # Key name must match eef_name used in FrankaPickBasketMimicEnv.target_eef_pose_to_action
        self.subtask_configs["franka"] = subtask_configs
