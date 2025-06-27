# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the Universal Robots.

The following configuration parameters are available:

* :obj:`UR10_CFG`: The UR10 arm without a gripper.

Reference: https://github.com/ros-industrial/universal_robot
"""
import os

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR

##
# Configuration
##


UR10_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAACLAB_NUCLEUS_DIR}/Robots/UniversalRobots/UR10/ur10_instanceable.usd",
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=5.0,
        ),
        activate_contact_sensors=False,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos={
            "shoulder_pan_joint": 0.0,
            "shoulder_lift_joint": -1.712,
            "elbow_joint": 1.712,
            "wrist_1_joint": 0.0,
            "wrist_2_joint": 0.0,
            "wrist_3_joint": 0.0,
        },
    ),
    actuators={
        "arm": ImplicitActuatorCfg(
            joint_names_expr=[".*"],
            velocity_limit=100.0,
            effort_limit=87.0,
            stiffness=800.0,
            damping=40.0,
        ),
    },
)

ur5_path = "gcu_objects/assets/ur_description/ur5e_instanceable.usd"
ur5_abs_path = os.path.abspath(ur5_path)

UR5_CFG = UR10_CFG.replace(
    spawn=sim_utils.UsdFileCfg(
        usd_path=ur5_abs_path,
    ),
)

ur5_robotiq_path = "gcu_objects/assets/ur_description/ur5e_robotiq_2f_140_temp.usd"
ur5_robotiq_abs_path = os.path.abspath(ur5_robotiq_path)

UR5_ROBOTIQ_CFG = UR5_CFG.replace(
    spawn=sim_utils.UsdFileCfg(
        usd_path=ur5_robotiq_abs_path,
    ),
)
"""Configuration of UR-10 arm using implicit actuator models."""
