# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

from . import agents

##
# Register Gym environments.
##

gym.register(
    id="Isaac-Pack-NoArm-v0",
    entry_point="gculab.envs:ManagerBasedRLGCUEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:NoArmPackEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:NoArmPackPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Pack-NoArm-Camera-v0",
    entry_point="gculab.envs:ManagerBasedRLGCUEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_pos_camera_env_cfg:NoArmPackCameraEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_camera_cfg:NoArmPackPPOCameraRunnerCfg",
    },
)

gym.register(
    id="Isaac-Pack-NoArm-Camera-Obj-Latent-v0",
    entry_point="gculab.envs:ManagerBasedRLGCUEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_pos_camera_env_cfg:NoArmPackCameraObjLatentEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_camera_obj_latent_cfg:NoArmPackPPOCameraObjLatentRunnerCfg",
    },
)
