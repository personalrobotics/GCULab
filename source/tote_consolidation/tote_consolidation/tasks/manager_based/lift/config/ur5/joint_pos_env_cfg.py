# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math

import tote_consolidation.tasks.manager_based.lift.mdp as mdp
from isaaclab.assets import ArticulationCfg
from isaaclab.controllers import DifferentialIKControllerCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import FrameTransformerCfg, OffsetCfg
from isaaclab.utils import configclass
from tote_consolidation.tasks.manager_based.lift.lift_env_cfg import LiftEnvCfg

##
# Pre-defined configs
##
from gculab_assets import UR5_ROBOTIQ_CFG  # isort: skip


##
# Environment configuration
##


@configclass
class UR5LiftEnvCfg(LiftEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        self.scene.right_robot = UR5_ROBOTIQ_CFG.replace(
            prim_path="{ENV_REGEX_NS}/RightRobot",
            init_state=ArticulationCfg.InitialStateCfg(
                pos=(0.9, 0.33, 0.75),
                rot=(0, -0.7071068, 0, 0.7071068),
                joint_pos={
                    "shoulder_pan_joint": 0.0,
                    "shoulder_lift_joint": -2.2,
                    "elbow_joint": 1.9,
                    "wrist_1_joint": -1.383,
                    "wrist_2_joint": -1.57,
                    "wrist_3_joint": 0.00,
                },
            ),
        )

        self.scene.left_robot = UR5_ROBOTIQ_CFG.replace(
            prim_path="{ENV_REGEX_NS}/LeftRobot",
            init_state=ArticulationCfg.InitialStateCfg(
                pos=(0.9, -0.33, 0.75),
                rot=(0, -0.7071068, 0, 0.7071068),
                joint_pos={
                    "shoulder_pan_joint": 0.0,
                    "shoulder_lift_joint": -2.2,
                    "elbow_joint": 1.9,
                    "wrist_1_joint": 1.383,
                    "wrist_2_joint": 1.57,
                    "wrist_3_joint": 0.00,
                },
            ),
        )

        # End-effector frame sensors for tracking gripper positions
        self.scene.ee_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/RightRobot/wrist_3_link",
            debug_vis=False,
            visualizer_cfg=None,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/RightRobot/wrist_3_link",
                    name="right_end_effector",
                    offset=OffsetCfg(pos=(0.0, 0.0, 0.1034)),  # offset to gripper tip
                ),
            ],
        )
        
        self.scene.left_ee_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/LeftRobot/wrist_3_link",
            debug_vis=False,
            visualizer_cfg=None,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/LeftRobot/wrist_3_link",
                    name="left_end_effector",
                    offset=OffsetCfg(pos=(0.0, 0.0, 0.1034)),  # offset to gripper tip
                ),
            ],
        )

        # override events
        # self.events.reset_robot_joints = EventTerm(
        #     func=mdp.reset_joints_by_scale,
        #     mode="reset",
        #     params={
        #         "position_range": (0.5, 1.5),
        #         "velocity_range": (0.0, 0.0),
        #     },
        # )
        # self.events.reset_robot_joints.params["position_range"] = (0.75, 1.25)
        # override rewards
        # self.rewards.end_effector_position_tracking.params["asset_cfg"].body_names = ["wrist_3_link"]
        # self.rewards.end_effector_position_tracking_fine_grained.params["asset_cfg"].body_names = ["wrist_3_link"]
        # self.rewards.end_effector_orientation_tracking.params["asset_cfg"].body_names = ["wrist_3_link"]
        # override actions
        # self.actions.arm_action = mdp.JointPositionActionCfg(
        #     asset_name="robot", joint_names=[".*"], scale=0.5, use_default_offset=True
        # )
        # override command generator body
        # end-effector is along x-direction
        # self.commands.ee_pose.body_name = "wrist_3_link"
        # self.commands.ee_pose.ranges.pitch = (math.pi / 2, math.pi / 2)

        # Set actions for both robots using SE3 actions for teleoperation compatibility
        # ORDER MATTERS: Teleop script expects [left_arm_action, arm_action, left_gripper_action, gripper_action]
        
        # Left robot actions (7 SE3 pose values)
        self.actions.left_arm_action = mdp.DifferentialInverseKinematicsActionCfg(
            asset_name="left_robot",
            joint_names=["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint", "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"],
            body_name="wrist_3_link", 
            controller=DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method="dls"),
        )
        
        # Right robot actions (7 SE3 pose values)  
        self.actions.right_arm_action = mdp.DifferentialInverseKinematicsActionCfg(
            asset_name="right_robot",
            joint_names=["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint", "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"],
            body_name="wrist_3_link",
            controller=DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method="dls"),
        )
        
        # Left gripper action (1 value)
        self.actions.left_gripper_action = mdp.BinaryJointPositionActionCfg(
            asset_name="left_robot",
            joint_names=["finger_joint"],
            open_command_expr={"finger_joint": 0.0},
            close_command_expr={"finger_joint": 0.8},
        )
        
        # Right gripper action (1 value)
        self.actions.right_gripper_action = mdp.BinaryJointPositionActionCfg(
            asset_name="right_robot",
            joint_names=["finger_joint"],
            open_command_expr={"finger_joint": 0.0},
            close_command_expr={"finger_joint": 0.8},
        )

        # Set the body name for the end effector (using right robot as primary for commands)
        # self.commands.object_pose.body_name = "wrist_3_link"

        # Add aliases for backward compatibility with teleop script
        # The teleop script expects "robot" and "object" entity names
        # We alias the right robot as "robot" for primary control
        self.scene.robot = self.scene.right_robot
        self.scene.object = self.scene.object0


@configclass
class UR5LiftEnvCfg_PLAY(UR5LiftEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # disable randomization for play
        self.observations.policy.enable_corruption = False
