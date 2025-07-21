# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to run an environment with a 3D Bin Packing agent based on the paper:
Stable bin packing of non-convex 3D objects with a robot manipulator
Fan Wang, Kris Hauser
https://arxiv.org/abs/1812.04093
"""

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
parser.add_argument("--exp_name", type=str, default="test_placement", help="Name of the experiment.")
parser.add_argument("--seed", type=int, default=0, help="Seed used for the environment")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import os
from datetime import datetime

import bpp_utils
import gymnasium as gym
import isaaclab.utils.math as math_utils
import isaaclab_tasks  # noqa: F401
import torch
import tote_consolidation.tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg

# PLACEHOLDER: Extension template (do not remove this comment)


def convert_transform_to_action_tensor(transforms, obj_indicies, device):
    """Convert a transform object to an action tensor format.

    Args:
        transforms: Transform object with position and attitude (orientation)
        obj_indicies: Index of the object to place
        device: The device to create tensors on

    Returns:
        A tensor representing the object index and transform in the format
        expected by the action space [obj_idx, pos_x, pos_y, pos_z, quat_w, quat_x, quat_y, quat_z]
    """
    # Get batch size from the transforms
    batch_size = len(transforms)
    action_tensor = torch.zeros((batch_size, 8), device=device)

    # Extract position and attitude values in batch
    positions = torch.tensor([[t.position.x, t.position.y, t.position.z] for t in transforms], device=device) / 100.0

    # Convert Euler angles to radians (vectorized)
    roll_rad = torch.tensor([t.attitude.roll for t in transforms], device=device) * torch.pi / 180.0
    pitch_rad = torch.tensor([t.attitude.pitch for t in transforms], device=device) * torch.pi / 180.0
    yaw_rad = torch.tensor([t.attitude.yaw for t in transforms], device=device) * torch.pi / 180.0

    # Convert Euler angles to quaternions (vectorized)
    quats = math_utils.quat_from_euler_xyz(roll_rad.unsqueeze(1), pitch_rad.unsqueeze(1), yaw_rad.unsqueeze(1)).squeeze(
        1
    )

    # Build action tensor
    action_tensor[:, 0] = obj_indicies
    action_tensor[:, 1:4] = positions
    action_tensor[:, 4:8] = quats

    return action_tensor

def main():
    """Zero actions agent with Isaac Lab environment."""
    # parse configuration
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    env_cfg.seed = args_cli.seed
    # create environment
    env = gym.make(args_cli.task, cfg=env_cfg)

    # print info (this is vectorized environment)
    print(f"[INFO]: Gym observation space: {env.observation_space}")
    print(f"[INFO]: Gym action space: {env.action_space}")
    # reset environment
    env.reset()

    obj_idx = torch.zeros(
        args_cli.num_envs, device=env.unwrapped.device, dtype=torch.int32
    )  # Track object indices per environment
    tote_manager = env.unwrapped.tote_manager
    num_obj_per_env = tote_manager.num_objects
    num_totes = len([key for key in env.unwrapped.scene.keys() if key.startswith("tote")])

    env_indices = torch.arange(args_cli.num_envs, device=env.unwrapped.device)  # Indices of all environments

    stats_dir = "stats"

    run_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    # create stats and run name directory if it does not exist
    if os.path.exists(stats_dir) is False:
        os.makedirs(stats_dir)
    run_name = f"{args_cli.task}_{run_dir}"
    # create run name directory
    run_path = os.path.join(stats_dir, run_name)
    if os.path.exists(run_path) is False:
        os.makedirs(run_path)
    exp_log_interval = 1  # Log stats every 50 steps

    step_count = 0

    bpp = bpp_utils.BPP(tote_manager, args_cli.num_envs, torch.arange(num_obj_per_env, device=env.unwrapped.device))

    while simulation_app.is_running():
        # run everything in inference mode
        with torch.inference_mode():
            # compute zero actions
            actions = torch.zeros(env.action_space.shape, device=env.unwrapped.device)
            stats = tote_manager.get_stats_summary()
            ejection_summary = tote_manager.stats.get_ejection_summary()
            print("GCU ", tote_manager.get_gcu(env_indices))
            print("\n===== Ejection Summary =====")
            print(f"Total steps: {stats['total_steps']}")
            if ejection_summary != {}:
                for i in range(len(ejection_summary.keys())):
                    env_id = list(ejection_summary.keys())[i]
                    print(ejection_summary[env_id])
                print("==========================\n")

            # [0] is destination tote idx (ascending values for batch size)
            # [1] currently is the object idx (0-indexed. -1 for no packable objects)
            # [2-9] is the desired object position and orientation
            # [10] is the action to indicate if an object is being placed
            actions[:, 0] = torch.arange(args_cli.num_envs, device=env.unwrapped.device) % num_totes

            tote_manager.eject_totes(actions[:, 0].to(torch.int32), env_indices)  # Eject destination totes

            # Destination tote IDs for each environment
            tote_ids = actions[:, 0].to(torch.int32)

            # Get the objects that can be packed
            packable_objects = bpp.get_packable_object_indices(num_obj_per_env, tote_manager, env_indices, tote_ids)[0]

            bpp.update_container_heightmap(env, env_indices, tote_ids)
            transforms, obj_indicies = bpp.get_action(env, packable_objects, tote_ids, env_indices)
            actions[:, 1:9] = convert_transform_to_action_tensor(transforms, obj_indicies, env.unwrapped.device)
            # apply actions
            env.step(actions)

            # Check that all environments have no packable objects
            tote_ids = actions[:, 0].to(torch.int32)  # Destination tote IDs for each environment

            if step_count % exp_log_interval == 0:
                print(f"\nStep {step_count}:")
                print("Saving stats to file...")
                tote_manager.stats.save_to_file(os.path.join(run_path, f"{args_cli.exp_name}.json"))
                print("Saved stats to file.")

            step_count += 1

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
