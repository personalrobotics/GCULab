# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to run an environment with zero action agent."""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Zero agent for Isaac Lab environments.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--visualize_spheres",
    action="store_true",
    help="When True, visualizes robot spheres",
    default=False,
)

parser.add_argument("--robot", type=str, default="ur5e_robotiq_2f_140.yml", help="robot configuration to load")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import carb
import gymnasium as gym
import isaaclab_tasks  # noqa: F401
import numpy as np
import omni.usd
import torch
from curobo.geom.sdf.world import CollisionCheckerType
from curobo.geom.types import WorldConfig
from curobo.types.base import TensorDeviceType
from curobo.types.math import Pose
from curobo.types.robot import JointState
from curobo.util.logger import setup_curobo_logger
from curobo.util.usd_helper import UsdHelper
from curobo.util_file import get_robot_configs_path, join_path, load_yaml
from curobo.wrap.reacher.ik_solver import IKSolver, IKSolverConfig
from isaaclab_tasks.utils import parse_env_cfg
from isaacsim.core.api.objects import cuboid, sphere

# Third Party
from omni.isaac.kit import SimulationApp


# CuRobo
def get_pose_grid(n_x, n_y, n_z, max_x, max_y, max_z):
    x = np.linspace(-max_x, max_x, n_x)
    y = np.linspace(-max_y, max_y, n_y)
    z = np.linspace(0, max_z, n_z)
    x, y, z = np.meshgrid(x, y, z, indexing="ij")

    position_arr = np.zeros((n_x * n_y * n_z, 3))
    position_arr[:, 0] = x.flatten()
    position_arr[:, 1] = y.flatten()
    position_arr[:, 2] = z.flatten()
    return position_arr


def draw_points(pose, success):
    # Third Party
    try:
        from omni.isaac.debug_draw import _debug_draw
    except ImportError:
        from isaacsim.util.debug_draw import _debug_draw

    draw = _debug_draw.acquire_debug_draw_interface()
    N = 100
    # if draw.get_num_points() > 0:
    draw.clear_points()
    cpu_pos = pose.position.cpu().numpy()
    b, _ = cpu_pos.shape
    point_list = []
    colors = []
    for i in range(b):
        # get list of points:
        point_list += [(cpu_pos[i, 0], cpu_pos[i, 1], cpu_pos[i, 2])]
        if success[i].item():
            colors += [(0, 1, 0, 0.25)]
        else:
            colors += [(1, 0, 0, 0.25)]
    sizes = [40.0 for _ in range(b)]

    draw.draw_points(point_list, colors, sizes)


# PLACEHOLDER: Extension template (do not remove this comment)


def main():
    """Zero actions agent with Isaac Lab environment."""
    # parse configuration
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    # create environment
    env = gym.make(args_cli.task, cfg=env_cfg)

    target = cuboid.VisualCuboid(
        "/World/target",
        position=np.array([0.5, 0, 0.05]),
        orientation=np.array([0, 1, 0, 0]),
        color=np.array([1.0, 0, 0]),
        size=0.05,
    )

    setup_curobo_logger("warn")
    past_pose = None
    n_obstacle_cuboids = 30
    n_obstacle_mesh = 10

    # warmup curobo instance
    usd_help = UsdHelper()
    target_pose = None

    tensor_args = TensorDeviceType()

    robot_cfg = load_yaml(join_path(get_robot_configs_path(), args_cli.robot))["robot_cfg"]

    j_names = robot_cfg["kinematics"]["cspace"]["joint_names"]
    default_config = robot_cfg["kinematics"]["cspace"]["retract_config"]

    robot_prim_path = "/World/envs/env_0/RightRobot"

    # Articulation
    robot = env.unwrapped.scene._articulations["right_robot"].root_physx_view

    world_cfg = WorldConfig()

    ik_config = IKSolverConfig.load_from_robot_config(
        robot_cfg,
        world_cfg,
        rotation_threshold=0.05,
        position_threshold=0.005,
        num_seeds=20,
        self_collision_check=True,
        self_collision_opt=True,
        tensor_args=tensor_args,
        use_cuda_graph=True,
        collision_checker_type=CollisionCheckerType.MESH,
        collision_cache={"obb": n_obstacle_cuboids, "mesh": n_obstacle_mesh},
        # use_fixed_samples=True,
    )
    ik_solver = IKSolver(ik_config)

    # get pose grid:
    position_grid_offset = tensor_args.to_device(get_pose_grid(10, 10, 5, 0.5, 0.5, 0.5))

    # read current ik pose and warmup?
    fk_state = ik_solver.fk(ik_solver.get_retract_config().view(1, -1))
    goal_pose = fk_state.ee_pose
    goal_pose = goal_pose.repeat(position_grid_offset.shape[0])
    goal_pose.position += position_grid_offset

    result = ik_solver.solve_batch(goal_pose)
    print("Curobo is Ready")

    usd_help.load_stage(omni.usd.get_context().get_stage())
    usd_help.add_world_to_stage(world_cfg, base_frame="/World")

    # print info (this is vectorized environment)
    print(f"[INFO]: Gym observation space: {env.observation_space}")
    print(f"[INFO]: Gym action space: {env.action_space}")
    # reset environment
    env.reset()
    step_index = 0
    cmd_plan = None
    spheres = None
    # compute zero actions
    actions = torch.zeros(env.action_space.shape, device=env.unwrapped.device)
    # simulate environment
    while simulation_app.is_running():
        # run everything in inference mode
        with torch.inference_mode():
            if step_index == 50 or step_index % 500 == 0.0:  # and cmd_plan is None:
                obstacles = usd_help.get_obstacles_from_stage(
                    only_paths=["/World"],
                    ignore_substring=[
                        robot_prim_path,
                        "/World/target",
                        "/World/defaultGroundPlane",
                        "/World/envs/env_0/Tote/visuals",
                        "/World/envs/env_0/Table/visuals",
                        "/World/envs/env_0/LeftRobot",  # Ignore left robot
                        "/curobo",
                    ],
                ).get_collision_check_world()
                print([x.name for x in obstacles.objects])
                ik_solver.update_world(obstacles)
                print("Updated World")
                carb.log_info("Synced CuRobo world from stage.")

            # position and orientation of target virtual cube:
            cube_position, cube_orientation = target.get_world_pose()

            if past_pose is None:
                past_pose = cube_position
            if target_pose is None:
                target_pose = cube_position
            sim_js_pos = robot.get_dof_positions()
            sim_js_vel = robot.get_dof_velocities()
            sim_js_names = robot.get_metatype(0).dof_names
            cu_js = JointState(
                position=tensor_args.to_device(sim_js_pos),
                velocity=tensor_args.to_device(sim_js_vel) * 0.0,
                acceleration=tensor_args.to_device(sim_js_vel) * 0.0,
                jerk=tensor_args.to_device(sim_js_vel) * 0.0,
                joint_names=sim_js_names,
            )
            cu_js = cu_js.get_ordered_joint_state(ik_solver.kinematics.joint_names)

            if args_cli.visualize_spheres and step_index % 2 == 0:
                sph_list = ik_solver.kinematics.get_robot_as_spheres(cu_js.position)

                if spheres is None:
                    spheres = []
                    # create spheres:

                    for si, s in enumerate(sph_list[0]):
                        sp = sphere.VisualSphere(
                            prim_path="/curobo/robot_sphere_" + str(si),
                            position=np.ravel(s.position),
                            radius=float(s.radius),
                            color=np.array([0, 0.8, 0.2]),
                        )
                        spheres.append(sp)
                else:
                    for si, s in enumerate(sph_list[0]):
                        spheres[si].set_world_pose(position=np.ravel(s.position))
                        spheres[si].set_radius(float(s.radius))
            if (
                np.linalg.norm(cube_position.cpu() - target_pose.cpu()) > 1e-3
                and np.linalg.norm(past_pose.cpu() - cube_position.cpu()) == 0.0
                and np.linalg.norm(sim_js_vel.cpu()) < 0.35
            ):
                print("Cube position changed, computing IK...")
                # Set EE teleop goals, use cube for simple non-vr init:
                ee_translation_goal = cube_position
                ee_orientation_teleop_goal = cube_orientation

                # compute curobo solution:
                ik_goal = Pose(
                    position=tensor_args.to_device(ee_translation_goal),
                    quaternion=tensor_args.to_device(ee_orientation_teleop_goal),
                )
                goal_pose.position[:] = ik_goal.position[:] + position_grid_offset
                goal_pose.quaternion[:] = ik_goal.quaternion[:]
                result = ik_solver.solve_batch(goal_pose)

                succ = torch.any(result.success)
                print("IK completed: Poses: " + str(goal_pose.batch) + " Time(s): " + str(result.solve_time))
                # get spheres and flags:
                draw_points(goal_pose, result.success)

                if succ:
                    # get all solutions:
                    cmd_plan = result.js_solution[result.success]
                    # get only joint names that are in both:
                    idx_list = []
                    common_js_names = []
                    for x in sim_js_names:
                        if x in cmd_plan.joint_names:
                            idx_list.append(robot.get_metatype(0).dof_indices[x])
                            common_js_names.append(x)
                    cmd_plan = cmd_plan.get_ordered_joint_state(common_js_names)

                    cmd_idx = 0

                else:
                    carb.log_warn("Plan did not converge to a solution.  No action is being taken.")
                target_pose = cube_position
            past_pose = cube_position
            if cmd_plan is not None and step_index % 20 == 0 and True:
                cmd_state = cmd_plan[cmd_idx]
                # Pad for gripper positions if necessary
                expected_size = robot.get_metatype(0).dof_count
                current_size = cmd_state.position.shape[0]
                if current_size < expected_size:
                    padded_position = torch.cat([
                        cmd_state.position,
                        torch.zeros(expected_size - current_size, device=cmd_state.position.device),
                    ])
                else:
                    padded_position = cmd_state.position

                robot.set_dof_positions(
                    padded_position, torch.tensor(idx_list, dtype=torch.int32, device=padded_position.device)
                )
                # set desired joint angles obtained from IK:
                # articulation_controller.apply_action(art_action)
                cmd_idx += 1
                if cmd_idx >= len(cmd_plan.position):
                    cmd_idx = 0
                    cmd_plan = None

                    # apply actions
                    robot.set_dof_positions(
                        torch.tensor(default_config, dtype=torch.float32, device=padded_position.device),
                        torch.tensor(idx_list, dtype=torch.int32, device=padded_position.device),
                    )
                else:
                    robot.set_dof_positions(
                        padded_position, torch.tensor(idx_list, dtype=torch.int32, device=padded_position.device)
                    )

            env.unwrapped.sim.render()
            step_index += 1

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
