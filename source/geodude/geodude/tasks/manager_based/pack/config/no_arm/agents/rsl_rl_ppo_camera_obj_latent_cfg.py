# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from gculab_rl.rsl_rl import RslRlPpoActorCriticConv2dPointNetCfg
from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg
from gculab_rl.rsl_rl import RslRlGCUPpoAlgorithmCfg

@configclass
class NoArmPackPPOCameraObjLatentRunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 4
    max_iterations = 3000
    save_interval = 10
    experiment_name = "no_arm_pack"
    empirical_normalization = True
    policy = RslRlPpoActorCriticConv2dPointNetCfg(
        init_noise_std=1.5,
        actor_hidden_dims=[128, 128],
        critic_hidden_dims=[128, 128],
        activation="elu",
        conv_layers_params=[
            {"out_channels": 4, "kernel_size": 3, "stride": 2, "padding": 1},
            {"out_channels": 16, "kernel_size": 3, "stride": 2},
        ],
        conv_linear_output_size=128,  # Project 128×13×10 into 256-dim
        pointnet_layers_params=[
            {"out_channels": 64},
            {"out_channels": 256},
        ],
        pointnet_in_dim=8,
        pointnet_num_points=512,
    )
    algorithm = RslRlGCUPpoAlgorithmCfg(
        value_loss_coef=0.5,
        use_clipped_value_loss=True,
        clip_param=0.2,
        placement_entropy_coef=0.0005,
        orientation_entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
