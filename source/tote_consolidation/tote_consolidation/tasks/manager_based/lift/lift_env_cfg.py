# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import os
from dataclasses import MISSING
import tempfile
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.devices.openxr import XrCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass

import isaaclab_tasks.manager_based.manipulation.lift.mdp as base_mdp

from . import mdp

tote_usd_path = "gcu_objects/assets/yellow_tote/model.usd"

vention_table_usd_path = "gcu_objects/assets/vention/vention.usd"

gcu_objects_path = os.path.abspath("../GCULab/gcu_objects")

num_object_per_env = 25
num_objects_to_reserve = 25

# Spacing between totes
tote_spacing = 0.43  # width of tote + gap between totes

##
# Scene definition
##


@configclass
class ObjectTableSceneCfg(InteractiveSceneCfg):
    """Configuration for the scene with a robotic arm."""

    # world
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -0.76)),
    )

    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        spawn=sim_utils.UsdFileCfg(
            usd_path=vention_table_usd_path,
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.45, 0.0, -0.0), rot=(0.70711, 0.0, 0.0, 0.70711)),
    )

    tote1 = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Tote1",
        spawn=sim_utils.UsdFileCfg(
            usd_path=tote_usd_path,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.45, -1.5 * tote_spacing, 0.0)),
    )

    tote2 = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Tote2",
        spawn=sim_utils.UsdFileCfg(
            usd_path=tote_usd_path,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.45, -0.5 * tote_spacing, 0.0)),
    )

    tote3 = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Tote3",
        spawn=sim_utils.UsdFileCfg(
            usd_path=tote_usd_path,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.45, 0.5 * tote_spacing, 0.0)),
    )

    tote4 = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Tote4",
        spawn=sim_utils.UsdFileCfg(
            usd_path=tote_usd_path,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.45, 1.5 * tote_spacing, 0.0)),
    )

    # robots
    right_robot: ArticulationCfg | None = None
    left_robot: ArticulationCfg | None = None

    # lights
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=2500.0),
    )

    def __post_init__(self):
        for i in range(num_object_per_env):
            setattr(
                self,
                f"object{i}",
                RigidObjectCfg(
                    prim_path=f"{{ENV_REGEX_NS}}/Object{i}",
                    spawn=sim_utils.MultiUsdFileCfg(
                        usd_path=[
                            os.path.join(gcu_objects_path, "YCB/Axis_Aligned_Physics/002_master_chef_can.usd"),
                            os.path.join(gcu_objects_path, "YCB/Axis_Aligned_Physics/003_cracker_box.usd"),
                            os.path.join(gcu_objects_path, "YCB/Axis_Aligned_Physics/004_sugar_box.usd"),
                            os.path.join(gcu_objects_path, "YCB/Axis_Aligned_Physics/005_tomato_soup_can.usd"),
                            os.path.join(gcu_objects_path, "YCB/Axis_Aligned_Physics/006_mustard_bottle.usd"),
                            # os.path.join(gcu_objects_path, "YCB/Axis_Aligned_Physics/007_tuna_fish_can.usd"),
                            # os.path.join(gcu_objects_path, "YCB/Axis_Aligned_Physics/008_pudding_box.usd"),
                            # os.path.join(gcu_objects_path, "YCB/Axis_Aligned_Physics/009_gelatin_box.usd"),
                            # os.path.join(gcu_objects_path, "YCB/Axis_Aligned_Physics/010_potted_meat_can.usd"),
                            # os.path.join(gcu_objects_path, "YCB/Axis_Aligned_Physics/011_banana.usd"),
                            # os.path.join(gcu_objects_path, "YCB/Axis_Aligned_Physics/019_pitcher_base.usd"),
                            # os.path.join(gcu_objects_path, "YCB/Axis_Aligned_Physics/021_bleach_cleanser.usd"),
                            # os.path.join(gcu_objects_path, "YCB/Axis_Aligned_Physics/024_bowl.usd"),
                            # os.path.join(gcu_objects_path, "YCB/Axis_Aligned_Physics/025_mug.usd"),
                            # os.path.join(gcu_objects_path, "YCB/Axis_Aligned_Physics/035_power_drill.usd"),
                            # os.path.join(gcu_objects_path, "YCB/Axis_Aligned_Physics/036_wood_block.usd"),
                            # os.path.join(gcu_objects_path, "YCB/Axis_Aligned_Physics/037_scissors.usd"),
                            # os.path.join(gcu_objects_path, "YCB/Axis_Aligned_Physics/040_large_marker.usd"),
                            # os.path.join(gcu_objects_path, "YCB/Axis_Aligned_Physics/051_large_clamp.usd"),
                            # os.path.join(gcu_objects_path, "YCB/Axis_Aligned_Physics/052_extra_large_clamp.usd"),
                            # os.path.join(gcu_objects_path, "YCB/Axis_Aligned_Physics/061_foam_brick.usd"),
                        ],
                        random_choice=True,
                        rigid_props=sim_utils.RigidBodyPropertiesCfg(
                            kinematic_enabled=False,
                            disable_gravity=False,
                            # enable_gyroscopic_forces=True,
                            solver_position_iteration_count=60,
                            solver_velocity_iteration_count=0,
                            sleep_threshold=0.005,
                            stabilization_threshold=0.0025,
                            max_depenetration_velocity=1000.0,
                        ),
                    ),
                    init_state=RigidObjectCfg.InitialStateCfg(pos=(i / 5.0, 1.2, -0.7)),
                ),
            )


##
# MDP settings
##


@configclass
class CommandsCfg:
    """Command terms for the MDP."""

    pass


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    # ORDER MATTERS: Teleop script expects [left_arm_action, arm_action, left_gripper_action, gripper_action]
    # Left arm actions (7 SE3 pose values)
    left_arm_action: mdp.JointPositionActionCfg | mdp.DifferentialInverseKinematicsActionCfg = MISSING
    
    # Right arm actions (7 SE3 pose values) 
    right_arm_action: mdp.JointPositionActionCfg | mdp.DifferentialInverseKinematicsActionCfg = MISSING
    
    # Left gripper actions (1 value)
    left_gripper_action: mdp.BinaryJointPositionActionCfg = MISSING
    
    # Right gripper actions (1 value)
    right_gripper_action: mdp.BinaryJointPositionActionCfg = MISSING


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group with state values."""

        actions = ObsTerm(func=mdp.last_action)
        
        # Robot state
        left_robot_joint_pos = ObsTerm(
            func=base_mdp.joint_pos,
            params={"asset_cfg": SceneEntityCfg("left_robot")},
        )
        right_robot_joint_pos = ObsTerm(
            func=base_mdp.joint_pos,
            params={"asset_cfg": SceneEntityCfg("right_robot")},
        )
        
        # Object observations
        object_pos = ObsTerm(func=base_mdp.root_pos_w, params={"asset_cfg": SceneEntityCfg("object")})
        object_rot = ObsTerm(func=base_mdp.root_quat_w, params={"asset_cfg": SceneEntityCfg("object")})
        
        # End-effector poses
        left_eef_pos = ObsTerm(func=mdp.get_left_eef_pos)
        left_eef_quat = ObsTerm(func=mdp.get_left_eef_quat)
        right_eef_pos = ObsTerm(func=mdp.get_right_eef_pos)
        right_eef_quat = ObsTerm(func=mdp.get_right_eef_quat)

        # Gripper states
        left_gripper_state = ObsTerm(func=mdp.get_left_gripper_state)
        right_gripper_state = ObsTerm(func=mdp.get_right_gripper_state)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    # """Configuration for events."""

    # obj_volume = EventTerm(
    #     func=mdp.object_obs,
    #     params={
    #         "asset_cfgs": [SceneEntityCfg(f"object{i}") for i in range(num_object_per_env)],
    #         "num_objects": num_object_per_env,
    #     },
    #     mode="startup",
    # )

    # reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")

    # randomize_objects = EventTerm(
    #     func=mdp.randomize_object_pose_with_invalid_ranges,
    #     params={
    #         "asset_cfgs": [SceneEntityCfg(f"object{i}") for i in range(num_object_per_env - num_objects_to_reserve)],
    #         "pose_range": {"x": (0.3, 0.5), "y": (-0.65, 0.65), "z": (0.6, 0.9)},
    #         "min_separation": 0.13,
    #         "invalid_ranges": [
    #             {"x": (0.3, 0.5), "y": (-0.07, 0.07)},  # center brim
    #             {"x": (0.3, 0.5), "y": (-tote_spacing - 0.07, -tote_spacing + 0.07)},  # left brim
    #             {"x": (0.3, 0.5), "y": (tote_spacing - 0.07, tote_spacing + 0.07)},  # right brim
    #         ],
    #     },
    #     mode="reset",
    # )

    # check_obj_out_of_bounds = EventTerm(
    #     func=mdp.check_obj_out_of_bounds,
    #     mode="post_reset",
    #     params={
    #         "asset_cfgs": [SceneEntityCfg(f"object{i}") for i in range(num_object_per_env - num_objects_to_reserve)],
    #     },
    # )

    # detect_objects_in_tote = EventTerm(
    #     func=mdp.detect_objects_in_tote,
    #     mode="post_reset",
    #     params={
    #         "asset_cfgs": [SceneEntityCfg(f"object{i}") for i in range(num_object_per_env - num_objects_to_reserve)],
    #     },
    # )

    # set_objects_to_invisible = EventTerm(
    #     func=mdp.set_objects_to_invisible,
    #     mode="post_reset",
    # )

    pass


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""
    
    pass


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    pass


@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""

    pass

@configclass
class ToteManagerCfg:
    num_object_per_env = num_object_per_env
    animate_vis = True


##
# Environment configuration
##


@configclass
class LiftEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the lifting environment."""

    # Scene settings
    scene: ObjectTableSceneCfg = ObjectTableSceneCfg(num_envs=1, env_spacing=2.5, replicate_physics=True)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    # MDP settings
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    tote_manager: ToteManagerCfg = ToteManagerCfg()

    # Unused managers
    commands = None
    rewards = None
    curriculum = None

    # Position of the XR anchor in the world frame
    xr: XrCfg = XrCfg(
        anchor_pos=(0.0, 0.0, 0.0),
        anchor_rot=(1.0, 0.0, 0.0, 0.0),
    )

        # Temporary directory for URDF files
    temp_urdf_dir = tempfile.gettempdir()

    idle_action = torch.tensor([
        # Left arm: [x, y, z, roll, pitch, yaw] - positioned to left side
        0.4, 0.3, 0.8, 0.0, 0.0, 0.0,
        # Right arm: [x, y, z, roll, pitch, yaw] - positioned to right side  
        0.4, -0.3, 0.8, 0.0, 0.0, 0.0,
        # Left gripper: open position
        1.0,
        # Right gripper: open position
        1.0,
    ])


    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 5
        self.episode_length_s = 20.0
        # simulation settings
        self.sim.dt = 1 / 60  # 60Hz
        self.sim.render_interval = 2

