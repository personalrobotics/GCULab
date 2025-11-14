# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math

import isaaclab_tasks.manager_based.manipulation.reach.mdp as mdp
from isaaclab.assets import ArticulationCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.utils import configclass
from geodude.tasks.manager_based.base.base_env_cfg import BaseEnvCfg

##
# Pre-defined configs
##
from gculab_assets import IMPLICIT_UR5_ROBOTIQ  # isort: skip

from geodude.tasks.manager_based.base import mdp

##
# Environment configuration
##

@configclass
class NoArmBaseEnvCfg(BaseEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

@configclass
class NoArmBaseEnvCfg_PLAY(NoArmBaseEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # disable randomization for play
        self.observations.policy.enable_corruption = False
