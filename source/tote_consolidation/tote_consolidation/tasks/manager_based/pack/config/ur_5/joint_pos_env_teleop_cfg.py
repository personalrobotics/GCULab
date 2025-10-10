# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math

import isaaclab_tasks.manager_based.manipulation.reach.mdp as mdp
from isaaclab.assets import ArticulationCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.utils import configclass
from tote_consolidation.tasks.manager_based.pack.config.ur_5.joint_pos_env_cfg import UR5PackEnvCfg
from tote_consolidation.tasks.manager_based.pack.pack_env_teleop_cfg import TeleopPackEnvCfg
from tote_consolidation.tasks.manager_based.pack.pack_env_cfg import PackEnvCfg

##
# Pre-defined configs
##
from gculab_assets import UR5_ROBOTIQ_CFG, IMPLICIT_UR5_ROBOTIQ, UR5_CFG  # isort: skip


##
# Environment configuration
##

@configclass
class UR5PackEnvTeleopCfg(TeleopPackEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        self.scene.right_robot = IMPLICIT_UR5_ROBOTIQ.replace(
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

        # self.scene.left_robot = UR5_ROBOTIQ_CFG.replace(
        #     prim_path="{ENV_REGEX_NS}/LeftRobot",
        #     init_state=ArticulationCfg.InitialStateCfg(
        #         pos=(0.9, -0.33, 0.75),
        #         rot=(0, -0.7071068, 0, 0.7071068),
        #         joint_pos={
        #             "shoulder_pan_joint": 0.0,
        #             "shoulder_lift_joint": -2.2,
        #             "elbow_joint": 1.9,
        #             "wrist_1_joint": 1.383,
        #             "wrist_2_joint": 1.57,
        #             "wrist_3_joint": 0.00,
        #         },
        #     ),
        # )

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
        self.actions.arm_action = mdp.JointPositionActionCfg(
            asset_name="right_robot", joint_names=["shoulder_pan_joint",
                "shoulder_lift_joint", 
                "elbow_joint",
                "wrist_1_joint",
                "wrist_2_joint", 
                "wrist_3_joint"], scale=1.0, use_default_offset= False
        )

        # self.actions.gripper_action = mdp.JointPositionActionCfg(
        #     asset_name="right_robot",
        #     joint_names=["finger_joint", "right_outer_knuckle_joint", "right_outer_finger_joint", "left_outer_finger_joint",
        #                  "left_inner_finger_pad_joint", "right_inner_finger_pad_joint",
        #                  "left_inner_finger_joint", "right_inner_finger_joint"],
        #     scale=1.0, use_default_offset=False,
        # )

        self.actions.gripper_action = mdp.BinaryJointPositionActionCfg(
               asset_name="right_robot",
            joint_names=["finger_joint", "right_outer_knuckle_joint", "right_outer_finger_joint", "left_outer_finger_joint",
                          "left_inner_finger_pad_joint", "right_inner_finger_pad_joint",
                         "left_inner_finger_joint", "right_inner_finger_joint"],
            open_command_expr={"finger_joint": 0.0, "right_outer_knuckle_joint": 0.0, "right_outer_finger_joint": 0.785398,
                "left_outer_finger_joint": 0.785398,
                "left_inner_finger_pad_joint": 0.0, "right_inner_finger_pad_joint": 0.0,
                "left_inner_finger_joint": -0.785398, "right_inner_finger_joint": -0.785398},
            close_command_expr={"finger_joint": 0.785398, "right_outer_knuckle_joint": 0.785398, "right_outer_finger_joint": 0.0,
                               "left_outer_finger_joint": 0.0,
                                 "left_inner_finger_pad_joint": 0.785398, "right_inner_finger_pad_joint": 0.785398,
                                    "left_inner_finger_joint": -0.785398, "right_inner_finger_joint": -0.785398},

            debug_vis=False,
        )


        # override command generator body
        # end-effector is along x-direction
        # self.commands.ee_pose.body_name = "wrist_3_link"
        # self.commands.ee_pose.ranges.pitch = (math.pi / 2, math.pi / 2)


@configclass
class UR5PackEnvTeleopCfg_PLAY(UR5PackEnvTeleopCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # disable randomization for play
        self.observations.policy.enable_corruption = False