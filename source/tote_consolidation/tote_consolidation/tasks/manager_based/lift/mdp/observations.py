# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations
import torch
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def get_left_eef_pos(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Get left UR5 end-effector position."""
    body_pos_w = env.scene["left_robot"].data.body_pos_w
    ee_idx = env.scene["left_robot"].data.body_names.index("wrist_3_link")  # UR5 end-effector
    eef_pos = body_pos_w[:, ee_idx] - env.scene.env_origins
    return eef_pos


def get_left_eef_quat(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Get left UR5 end-effector quaternion."""
    body_quat_w = env.scene["left_robot"].data.body_quat_w
    ee_idx = env.scene["left_robot"].data.body_names.index("wrist_3_link")
    eef_quat = body_quat_w[:, ee_idx]
    return eef_quat


def get_right_eef_pos(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Get right UR5 end-effector position."""
    body_pos_w = env.scene["right_robot"].data.body_pos_w
    ee_idx = env.scene["right_robot"].data.body_names.index("wrist_3_link")
    eef_pos = body_pos_w[:, ee_idx] - env.scene.env_origins
    return eef_pos


def get_right_eef_quat(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Get right UR5 end-effector quaternion."""
    body_quat_w = env.scene["right_robot"].data.body_quat_w
    ee_idx = env.scene["right_robot"].data.body_names.index("wrist_3_link")
    eef_quat = body_quat_w[:, ee_idx]
    return eef_quat


def get_left_gripper_state(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Get left gripper joint positions."""
    robot_joint_names = env.scene["left_robot"].data.joint_names
    gripper_joints = ["finger_joint"]  # UR5 gripper joints
    indexes = torch.tensor([robot_joint_names.index(name) for name in gripper_joints], dtype=torch.long)
    return env.scene["left_robot"].data.joint_pos[:, indexes]


def get_right_gripper_state(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Get right gripper joint positions."""
    robot_joint_names = env.scene["right_robot"].data.joint_names
    gripper_joints = ["finger_joint"]  # UR5 gripper joints
    indexes = torch.tensor([robot_joint_names.index(name) for name in gripper_joints], dtype=torch.long)
    return env.scene["right_robot"].data.joint_pos[:, indexes]


def object_obs(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Object observations relative to both arms."""
    # Get both end-effector positions
    left_eef_pos = get_left_eef_pos(env)
    right_eef_pos = get_right_eef_pos(env)
    
    # Object state
    object_pos = env.scene["object"].data.root_pos_w - env.scene.env_origins
    object_quat = env.scene["object"].data.root_quat_w
    
    # Relative positions
    left_eef_to_object = object_pos - left_eef_pos
    right_eef_to_object = object_pos - right_eef_pos
    
    return torch.cat([object_pos, object_quat, left_eef_to_object, right_eef_to_object], dim=1)
