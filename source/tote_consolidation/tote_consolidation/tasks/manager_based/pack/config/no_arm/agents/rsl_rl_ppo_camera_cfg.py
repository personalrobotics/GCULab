# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from gculab_rl.rsl_rl import RslRlPpoActorCriticConv2dCfg
from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg


@configclass
class NoArmPackPPOCameraRunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 1500
    save_interval = 10
    experiment_name = "no_arm_pack"
    empirical_normalization = True
    policy = RslRlPpoActorCriticConv2dCfg(
        init_noise_std=40.0,        
        actor_hidden_dims=[256, 128],
        critic_hidden_dims=[256, 128],
        activation="elu",
        conv_layers_params=[
            {"out_channels": 4, "kernel_size": 3, "stride": 2, "padding": 1},
            {"out_channels": 8, "kernel_size": 3, "stride": 2},
            {"out_channels": 16, "kernel_size": 3, "stride": 2},
        ],
        conv_linear_output_size=128,  # Project 128×13×10 into 128-dim
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=0.5,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.0025,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )