# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

##
# Scene definition
##
import os
from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import ActionTermCfg as ActionTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from . import mdp

tote_usd_path = "gcu_objects/assets/yellow_tote/model.usd"
tote_usd_abs_path = os.path.abspath(tote_usd_path)

vention_table_usd_path = "gcu_objects/assets/vention/vention.usd"

num_object_per_env = 5


@configclass
class PackSceneCfg(InteractiveSceneCfg):
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

    tote = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Tote",
        spawn=sim_utils.UsdFileCfg(
            usd_path=tote_usd_abs_path,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.45, 0.0, 0.0)),
    )

    tote2 = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Tote2",
        spawn=sim_utils.UsdFileCfg(
            usd_path=tote_usd_abs_path,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.45, 0.42, 0.0)),
    )

    tote3 = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Tote3",
        spawn=sim_utils.UsdFileCfg(
            usd_path=tote_usd_abs_path,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.45, -0.42, 0.0)),
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
                f"object{i+1}",
                RigidObjectCfg(
                    prim_path=f"{{ENV_REGEX_NS}}/Object{i+1}",
                    spawn=sim_utils.MultiUsdFileCfg(
                        usd_path=[
                            "/home/henri/Downloads/YCB/Axis_Aligned_Physics/002_master_chef_can.usd",
                            "/home/henri/Downloads/YCB/Axis_Aligned_Physics/003_cracker_box.usd",
                            "/home/henri/Downloads/YCB/Axis_Aligned_Physics/004_sugar_box.usd",
                            "/home/henri/Downloads/YCB/Axis_Aligned_Physics/005_tomato_soup_can.usd",
                            "/home/henri/Downloads/YCB/Axis_Aligned_Physics/006_mustard_bottle.usd",
                            "/home/henri/Downloads/YCB/Axis_Aligned_Physics/007_tuna_fish_can.usd",
                            "/home/henri/Downloads/YCB/Axis_Aligned_Physics/008_pudding_box.usd",
                            "/home/henri/Downloads/YCB/Axis_Aligned_Physics/009_gelatin_box.usd",
                            "/home/henri/Downloads/YCB/Axis_Aligned_Physics/010_potted_meat_can.usd",
                            "/home/henri/Downloads/YCB/Axis_Aligned_Physics/011_banana.usd",
                            "/home/henri/Downloads/YCB/Axis_Aligned_Physics/019_pitcher_base.usd",
                            "/home/henri/Downloads/YCB/Axis_Aligned_Physics/021_bleach_cleanser.usd",
                            "/home/henri/Downloads/YCB/Axis_Aligned_Physics/024_bowl.usd",
                            "/home/henri/Downloads/YCB/Axis_Aligned_Physics/025_mug.usd",
                            "/home/henri/Downloads/YCB/Axis_Aligned_Physics/035_power_drill.usd",
                            "/home/henri/Downloads/YCB/Axis_Aligned_Physics/036_wood_block.usd",
                            "/home/henri/Downloads/YCB/Axis_Aligned_Physics/037_scissors.usd",
                            "/home/henri/Downloads/YCB/Axis_Aligned_Physics/040_large_marker.usd",
                            "/home/henri/Downloads/YCB/Axis_Aligned_Physics/051_large_clamp.usd",
                            "/home/henri/Downloads/YCB/Axis_Aligned_Physics/052_extra_large_clamp.usd",
                            "/home/henri/Downloads/YCB/Axis_Aligned_Physics/061_foam_brick.usd",
                        ],
                        random_choice=True,
                        rigid_props=sim_utils.RigidBodyPropertiesCfg(
                            kinematic_enabled=False,
                            disable_gravity=True,
                        ),
                        # collision_props=sim_utils.CollisionPropertiesCfg(
                        #     collision_enabled=False,
                        # ),
                    ),
                    init_state=RigidObjectCfg.InitialStateCfg(pos=(i / 5.0, 0.0, 2.0)),
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

    arm_action: ActionTerm | None = None
    gripper_action: ActionTerm | None = None
    packing_action: mdp.PackingAction | None = None


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Configuration for events."""

    obj_volume = EventTerm(
        func=mdp.object_props,
        params={
            "asset_cfgs": [SceneEntityCfg(f"object{i + 1}") for i in range(num_object_per_env)],
            "num_objects": num_object_per_env,
        },
        mode="startup",
    )

    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    pass


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""


@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""

    pass


@configclass
class GCUCfg:
    num_object_per_env = num_object_per_env


##
# Environment configuration
##


@configclass
class PackEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the reach end-effector pose tracking environment."""

    # Scene settings
    scene: PackSceneCfg = PackSceneCfg(num_envs=4096, env_spacing=2.5, replicate_physics=False)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()
    gcu: GCUCfg = GCUCfg()

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 2
        self.sim.render_interval = self.decimation
        self.episode_length_s = 10.0
        self.viewer.eye = (3.5, 3.5, 3.5)
        # simulation settings
        self.sim.dt = 1.0 / 60.0
