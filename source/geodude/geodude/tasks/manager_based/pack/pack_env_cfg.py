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
from isaaclab.managers import RewardTermCfg as RewardTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass
from geodude.tasks.manager_based.base.base_env_cfg import BaseSceneCfg

from . import mdp

tote_usd_path = "gcu_objects/assets/yellow_tote/model.usd"

vention_table_usd_path = "gcu_objects/assets/vention/vention.usd"

gcu_objects_path = os.path.abspath("gcu_objects")

# Dynamically build list of USD paths by scanning the directory
ycb_physics_dir = os.path.join(gcu_objects_path, "YCB/Axis_Aligned_Physics")
lw_physics_dir = os.path.join(gcu_objects_path, "YCB/lightwheel_usds/gcu_assets")
ycb_usd_files = glob.glob(os.path.join(ycb_physics_dir, "*.usd"))
lw_usd_files = glob.glob(os.path.join(lw_physics_dir, "*.usd"))

# Extract all available object IDs and names for reference
available_objects = {}
for usd_file in ycb_usd_files:
    basename = os.path.basename(usd_file)
    obj_id = basename[:3]
    obj_name = basename[4:].replace(".usd", "")
    available_objects[obj_id] = obj_name

# Print available objects for reference
print("Available YCB objects:")
for obj_id, obj_name in sorted(available_objects.items()):
    print(f'"{obj_id}", # {obj_name}')

# Define which object IDs to include
ycb_include_ids = [
    "003",  # cracker_box
    # "004",  # sugar_box
    # "006",  # mustard_bottle
    # "007",  # tuna_fish_can
    # "008",  # pudding_box
    # "009",  # gelatin_box
    "010", # potted_meat_can
    # "011",  # banana
    # "024", # bowl
    # "025", # mug
    # "036",  # wood_block
    # "051", # large_clamp
    # "052", # extra_large_clamp
    # "061",  # foam_brick
]

lw_include_names = [
    # "cracker_box",
    # "bowl",
]

# Filter USD files based on ID prefixes
usd_paths = []
for usd_file in ycb_usd_files:
    basename = os.path.basename(usd_file)
    # Extract the 3-digit ID from filename (assuming format like "003_cracker_box.usd")
    if basename[:3] in ycb_include_ids:
        usd_paths.append(usd_file)

for usd_file in lw_usd_files:
    basename = os.path.basename(usd_file)
    base_name = basename.replace(".usd", "")
    if base_name in lw_include_names:
        usd_paths.append(usd_file)

num_object_per_env = 70

# Spacing between totes
tote_spacing = 0.43  # width of tote + gap between totes


@configclass
class PackSceneCfg(BaseSceneCfg):
    """Configuration for the scene with totes."""
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

    def __post_init__(self):
        super().__post_init__()

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
                            solver_position_iteration_count=4,
                            solver_velocity_iteration_count=0,
                            sleep_threshold=0.005,
                            stabilization_threshold=0.0025,
                            # max_depenetration_velocity=1000.0,
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
        # actions = ObsTerm(func=mdp.last_action)
        obs_dims = ObsTerm(func=mdp.obs_dims)

        def __post_init__(self):
            self.enable_corruption = True

        #     self.concatenate_terms = True

    class SensorCfg(ObsGroup):
        """Observations for sensor group."""

        heightmap = ObsTerm(func=mdp.heightmap)

    # observation groups
    policy: PolicyCfg = PolicyCfg()
    sensor: SensorCfg = SensorCfg()


@configclass
class EventCfg:
    """Configuration for events."""

    obj_volume = EventTerm(
        func=mdp.object_props,
        params={
            "asset_cfgs": [SceneEntityCfg(f"object{i}") for i in range(num_object_per_env)],
            "num_objects": num_object_per_env,
        },
        mode="startup",
    )

    refill_source_totes = EventTerm(
        func=mdp.refill_source_totes,
        mode="reset",
    )

    set_objects_to_invisible = EventTerm(
        func=mdp.set_objects_to_invisible,
        mode="post_reset",
    )

    log_gcu_dest_tote = EventTerm(
        func=mdp.log_gcu_dest_tote,
        mode="interval",
        interval_range_s=(0.0, 0.0),
    )

    log_gcu_max = EventTerm(
        func=mdp.log_gcu_max,
        mode="interval",
        interval_range_s=(0.0, 0.0),
    )


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    gcu_reward = RewardTerm(
        func=mdp.gcu_reward_step, weight=1000.0
    )

    object_shift = RewardTerm(func=mdp.object_shift, weight=10.0)

    wasted_volume = RewardTerm(func=mdp.wasted_volume_pbrs, weight=40.0)

@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    object_overfilled_tote = DoneTerm(
        func=mdp.object_overfilled_tote,
    )


@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""

    pass
    # object_shift = CurriculumTerm(
    #     func=mdp.modify_reward_weight, params={"term_name": "object_shift", "weight": 50.0, "num_steps": 10000}
    # )


@configclass
class ToteManagerCfg:
    num_object_per_env = num_object_per_env
    animate_vis = False
    obj_settle_wait_steps = 50
    disable_logging: bool = False


##
# Environment configuration
##


@configclass
class PackEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the reach end-effector pose tracking environment."""

    # Scene settings
    scene: PackSceneCfg = PackSceneCfg(num_envs=512, env_spacing=2.5, replicate_physics=False)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()
    tote_manager: ToteManagerCfg = ToteManagerCfg()

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 1
        self.sim.render_interval = self.decimation
        self.episode_length_s = 10.0
        self.viewer.eye = (0, 0.1, 5.5)
        # simulation settings
        self.sim.dt = 1.0 / 90.0
        self.sim.physx.gpu_max_rigid_patch_count = 4096 * 4096
        self.sim.physx.gpu_collision_stack_size = 4096 * 4096 * 20
        self.sim.physx.gpu_found_lost_pairs_capacity = 4096 * 4096 * 20
        self.sim.physx.gpu_max_rigid_contact_count = 2**26
