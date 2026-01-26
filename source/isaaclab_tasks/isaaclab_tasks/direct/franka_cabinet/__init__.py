# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""
Franka-Cabinet environment.
"""

import gymnasium as gym

from . import agents

##
# Register Gym environments.
##

gym.register(
    id="Isaac-Franka-Cabinet-Direct-v0",
    entry_point=f"{__name__}.franka_cabinet_env:FrankaCabinetEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.franka_cabinet_env:FrankaCabinetEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:FrankaCabinetPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Franka-Test-Direct-v0",
    entry_point=f"{__name__}.franka_test_env:FrankaTestEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.franka_test_env:FrankaTestEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:FrankaTestPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)


# living_room_scene1_pick_up_the_cream_cheese_box_and_put_it_in_the_basket
gym.register(
    id="LivingRoomScene1PickUpTheCreamCheeseAndPutItInTheBasket-v0",
    entry_point=f"{__name__}.living_room_scene1_pick_up_the_cream_cheese_box_and_put_it_in_the_basket:LivingRoomScene1PickUpTheCreamCheeseAndPutItInTheBasket",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.living_room_scene1_pick_up_the_cream_cheese_box_and_put_it_in_the_basket:LivingRoomScene1PickUpTheCreamCheeseAndPutItInTheBasketCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:FrankaTestPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)

gym.register(
    id="ReplayLivingRoomScene1PickUpTheAlphabetSoupAndPutItInTheBasket",
    entry_point=f"{__name__}.replay_living_room_scene1_pick_up_the_alphabet_soup_and_put_it_in_the_basket:ReplayLivingRoomScene1PickUpTheAlphabetSoupAndPutItInTheBasket",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.replay_living_room_scene1_pick_up_the_alphabet_soup_and_put_it_in_the_basket:ReplayLivingRoomScene1PickUpTheAlphabetSoupAndPutItInTheBasketCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:FrankaTestPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)

gym.register(
    id="LivingRoomScene1PickUpTheAlphabetSoupAndPutItInTheBasket",
    entry_point=f"{__name__}.living_room_scene1_pick_up_the_alphabet_soup_and_put_it_in_the_basket:LivingRoomScene1PickUpTheAlphabetSoupAndPutItInTheBasket",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.living_room_scene1_pick_up_the_alphabet_soup_and_put_it_in_the_basket:LivingRoomScene1PickUpTheAlphabetSoupAndPutItInTheBasketCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:FrankaTestPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)