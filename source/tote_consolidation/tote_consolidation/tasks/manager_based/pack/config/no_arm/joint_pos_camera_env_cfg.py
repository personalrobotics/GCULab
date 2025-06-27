# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass
from tote_consolidation.tasks.manager_based.pack.pack_camera_env_cfg import (
    PackResNet18DepthCameraEnvCfg,
)

##
# Environment configuration
##


@configclass
class NoArmPackCameraEnvCfg(PackResNet18DepthCameraEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # switch robot to franka
        self.scene.robot = None


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
