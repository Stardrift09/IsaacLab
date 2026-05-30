# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

gym.register(
    id="Isaac-Franka-PickBasket-IK-Rel-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.pick_basket_ik_rel_env_cfg:FrankaPickBasketEnvCfg",
    },
    disable_env_checker=True,
)

