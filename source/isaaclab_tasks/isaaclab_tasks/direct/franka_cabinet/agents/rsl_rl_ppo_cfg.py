# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


@configclass
class FrankaCabinetPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 16
    max_iterations = 1500
    save_interval = 50
    experiment_name = "franka_cabinet_direct"
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_obs_normalization=True,
        critic_obs_normalization=True,
        actor_hidden_dims=[256, 128, 64],
        critic_hidden_dims=[256, 128, 64],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.0,
        num_learning_epochs=8,
        num_mini_batches=8,
        learning_rate=5.0e-4,
        schedule="adaptive",
        gamma=0.9,
        lam=0.95,
        desired_kl=0.008,
        max_grad_norm=1.0,
    )

@configclass
class FrankaTestPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 16
    max_iterations = 1500
    save_interval = 50
    experiment_name = "franka_test_direct"
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_obs_normalization=True,
        critic_obs_normalization=True,
        actor_hidden_dims=[256, 128, 64],
        critic_hidden_dims=[256, 128, 64],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.0,
        num_learning_epochs=8,
        num_mini_batches=8,
        learning_rate=5.0e-4,
        schedule="adaptive",
        gamma=0.9, # 0.9 works 
        lam=0.95,
        desired_kl=0.008,
        max_grad_norm=1.0,
    )
    

@configclass
class TestPutItInTheBasketPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 16
    max_iterations = 1500
    save_interval = 50
    experiment_name = "test_put_it_in_the_basket"
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_obs_normalization=True,
        critic_obs_normalization=True,
        actor_hidden_dims=[256, 128, 64],
        critic_hidden_dims=[256, 128, 64],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.0,
        num_learning_epochs=8,
        num_mini_batches=8,
        learning_rate=5.0e-4,
        schedule="adaptive",
        gamma=0.95, # 0.9 works for picking up the object already
        lam=0.95,
        desired_kl=0.008,
        max_grad_norm=1.0,
    )


@configclass
class TestPickItUpPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 16
    max_iterations = 1500
    save_interval = 50
    experiment_name = "test_pick_it_up"
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_obs_normalization=True,
        critic_obs_normalization=True,
        actor_hidden_dims=[256, 128, 64],
        critic_hidden_dims=[256, 128, 64],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.0,
        num_learning_epochs=8,
        num_mini_batches=8,
        learning_rate=5.0e-4,
        schedule="adaptive",
        gamma=0.95,
        lam=0.95,
        desired_kl=0.008,
        max_grad_norm=1.0,
    )

@configclass
class TestStageAsFeedbackPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 16
    max_iterations = 1500
    save_interval = 50
    experiment_name = "test_stage_as_feedback"
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_obs_normalization=True,
        critic_obs_normalization=True,
        actor_hidden_dims=[256, 128, 64],
        critic_hidden_dims=[256, 128, 64],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.0,
        num_learning_epochs=8,
        num_mini_batches=8,
        learning_rate=5.0e-4,
        schedule="adaptive",
        gamma=0.95,
        lam=0.95,
        desired_kl=0.008,
        max_grad_norm=1.0,
    )