# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import geodude.tasks.manager_based.pack.mdp as mdp
from isaaclab.utils import configclass
from geodude.tasks.manager_based.pack.pack_camera_env_cfg import (
    PackDepthCameraEnvCfg,
    PackDepthCameraObjLatentEnvCfg,
)

##
# Environment configuration
##


@configclass
class NoArmPackCameraEnvCfg(PackDepthCameraEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        self.scene.robot = None
        self.actions.packing_action = mdp.PackingActionCfg(
            asset_name="tote1", place_obj_bottomLeft=True
        )  # asset name is not used in this env


@configclass
class NoArmPackCameraObjLatentEnvCfg(PackDepthCameraObjLatentEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        self.scene.robot = None
        self.actions.packing_action = mdp.PackingActionCfg(
            asset_name="tote1", place_obj_bottomLeft=True
        )  # asset name is not used in this env


@configclass
class NoArmPackCameraEnvCfg_PLAY(NoArmPackCameraEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # disable randomization for play
        self.observations.policy.enable_corruption = False
