# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

##
# Register Gym environments.
##

gym.register(
    id="Isaac-Pack-UR5-v0",
    entry_point="gculab.envs:ManagerBasedRLGCUEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:UR5PackEnvCfg",
    },
)
