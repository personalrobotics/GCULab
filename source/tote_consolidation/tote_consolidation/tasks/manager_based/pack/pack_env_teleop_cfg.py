# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import glob

##
# Scene definition
##
import os
from dataclasses import MISSING

import gculab.sim as gcu_sim_utils
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
from tote_consolidation.tasks.manager_based.pack.pack_env_cfg import ActionsCfg, CommandsCfg, CurriculumCfg, EventCfg, ObservationsCfg, PackEnvCfg, PackSceneCfg, RewardsCfg, TerminationsCfg, ToteManagerCfg

from . import mdp

tote_usd_path = "gcu_objects/assets/yellow_tote/model.usd"

vention_table_usd_path = "gcu_objects/assets/vention/vention.usd"

gcu_objects_path = os.path.abspath("gcu_objects")

# Dynamically build list of USD paths by scanning the directory
ycb_physics_dir = os.path.join(gcu_objects_path, "YCB/Axis_Aligned_Physics")
all_usd_files = glob.glob(os.path.join(ycb_physics_dir, "*.usd"))

# Extract all available object IDs and names for reference
available_objects = {}
for usd_file in all_usd_files:
    basename = os.path.basename(usd_file)
    obj_id = basename[:3]
    obj_name = basename[4:].replace(".usd", "")
    available_objects[obj_id] = obj_name

# Print available objects for reference
print("Available YCB objects:")
for obj_id, obj_name in sorted(available_objects.items()):
    print(f'"{obj_id}", # {obj_name}')

# Define which object IDs to include
include_ids = [
    "003",  # cracker_box
    "004",  # sugar_box
    "006",  # mustard_bottle
    "008",  # pudding_box
    "009",  # gelatin_box
    "036",  # wood_block
    "061",  # foam_brick
]

# Filter USD files based on ID prefixes
usd_paths = []
for usd_file in all_usd_files:
    basename = os.path.basename(usd_file)
    # Extract the 3-digit ID from filename (assuming format like "003_cracker_box.usd")
    if basename[:3] in include_ids:
        usd_paths.append(usd_file)

num_object_per_env = 50
num_objects_to_reserve = 50

# Spacing between totes
tote_spacing = 0.43  # width of tote + gap between totes


@configclass
class PackSceneTeleopCfg(InteractiveSceneCfg):
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
                    spawn=gcu_sim_utils.MultiUsdFromDistFileCfg(
                        usd_path=usd_paths,
                        random_choice=True,
                        distribution=None,  # None for uniform distribution
                        rigid_props=sim_utils.RigidBodyPropertiesCfg(
                            kinematic_enabled=False,
                            disable_gravity=False,
                            # enable_gyroscopic_forces=True,
                            solver_position_iteration_count=90,
                            solver_velocity_iteration_count=0,
                            sleep_threshold=0.005,
                            stabilization_threshold=0.0025,
                            max_depenetration_velocity=1000.0,
                        ),
                    ),
                    init_state=RigidObjectCfg.InitialStateCfg(pos=(i / 5.0, 1.2, -0.7)),
                ),
            )


@configclass
class TeleopEventCfg:
    """Configuration for events."""
    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")

##
# Environment configuration
##


@configclass
class TeleopPackEnvCfg(PackEnvCfg):
    """Configuration for the reach end-effector pose tracking environment."""

    # Scene settings
    scene: PackSceneTeleopCfg = PackSceneTeleopCfg(num_envs=4096, env_spacing=2.5, replicate_physics=False)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: TeleopEventCfg = TeleopEventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()
    tote_manager: ToteManagerCfg = ToteManagerCfg()

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 2
        self.sim.render_interval = self.decimation
        self.episode_length_s = 10.0
        self.viewer.eye = (0, 0.1, 5.5)
        # simulation settings
        self.sim.dt = 1.0 / 90.0
