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
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import isaaclab_tasks  # noqa: F401
import torch
from isaaclab_tasks.utils import parse_env_cfg

import tote_consolidation.tasks  # noqa: F401


# PLACEHOLDER: Extension template (do not remove this comment)
def get_packable_object_indices(num_obj_per_env, tote_manager, env_indices, tote_ids):
    """Get indices of objects that can be packed per environment.

    Args:
        num_obj_per_env: Number of objects per environment
        tote_manager: The tote manager object
        env_indices: Indices of environments to get packable objects for
        tote_ids: Destination tote IDs for each environment

    Returns:
        List of tensors containing packable object indices for each environment
    """
    num_envs = env_indices.shape[0]

    # Get objects that are reserved (already being picked up)
    reserved_objs = tote_manager.get_reserved_objs_idx(env_indices)

    # Get objects that are already in destination totes
    objs_in_dest = tote_manager.get_tote_objs_idx(tote_ids, env_indices)

    # Create a 2D tensor of object indices: shape (num_envs, num_obj_per_env)
    obj_indices = torch.arange(1, num_obj_per_env + 1, device=env_indices.device).expand(num_envs, -1)

    # Compute mask of packable objects
    mask = (~reserved_objs & ~objs_in_dest).bool()

    # Use list comprehension to get valid indices per environment
    valid_indices = [obj_indices[i][mask[i]] for i in range(num_envs)]

    return valid_indices, mask


def select_random_packable_objects(packable_objects, packable_mask, device, num_obj_per_env):
    """Select random packable objects for each environment.

    Args:
        packable_objects: List of tensors with packable object indices for each environment
        packable_mask: Boolean mask of packable objects
        device: Device to create tensors on
        num_obj_per_env: Number of objects per environment

    Returns:
        Tensor of selected object indices (zeros for environments with no packable objects)
    """
    num_envs = len(packable_objects)
    selected_obj_indices = torch.zeros(num_envs, device=device, dtype=torch.int32)

    # Count available objects per environment
    available_obj_counts = packable_mask.sum(dim=1)
    has_objects = available_obj_counts > 0

    # Early return if no objects available
    if not has_objects.any():
        return selected_obj_indices

    # Pad all objects into a single tensor
    all_objects = torch.zeros(num_envs, num_obj_per_env, device=device)
    for i, objs in enumerate(packable_objects):
        if len(objs) > 0:
            all_objects[i, : len(objs)] = objs

    # Sample random indices based on available objects
    random_values = torch.rand(num_envs, device=device)
    random_indices = (random_values * available_obj_counts).to(torch.int64)

    # Select objects using vectorized indexing - fix the broadcasting issue
    # We need to explicitly select each environment with its corresponding index
    sampled_objects = all_objects[torch.arange(num_envs, device=device), random_indices]

    # Apply mask for environments with objects
    selected_obj_indices = torch.where(has_objects, sampled_objects.to(torch.int32), selected_obj_indices)

    return selected_obj_indices


def main():
    """Zero actions agent with Isaac Lab environment."""
    # parse configuration
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    # create environment
    env = gym.make(args_cli.task, cfg=env_cfg)

    # print info (this is vectorized environment)
    print(f"[INFO]: Gym observation space: {env.observation_space}")
    print(f"[INFO]: Gym action space: {env.action_space}")
    # reset environment
    env.reset()

    idx = 0
    obj_idx = torch.ones(
        env.num_envs, device=env.unwrapped.device, dtype=torch.int32
    )  # Track object indices per environment
    tote_manager = env.unwrapped.tote_manager
    num_obj_per_env = tote_manager.num_objects

    # Add counters to track packing order for drop height calculation
    packing_counters = torch.zeros(env.num_envs, device=env.unwrapped.device, dtype=torch.int32)

    # simulate environment
    while simulation_app.is_running():
        # run everything in inference mode
        with torch.inference_mode():
            # compute zero actions
            actions = torch.zeros(env.action_space.shape, device=env.unwrapped.device)
            if idx % 50 == 0 and idx != 0:
                batch_size = actions.shape[0]
                num_totes = len([key for key in env.unwrapped.scene.keys() if key.startswith("tote")])
                # [0] is destination tote idx (ascending values for batch size)
                # [1] currently is the object idx (1-indexed. Set to 0 for no object selected)
                # [2-9] is the desired object position and orientation
                actions[:, 0] = torch.arange(batch_size, device=env.unwrapped.device) % num_totes

                # Get the tote IDs and environment indices
                tote_ids = actions[:, 0].to(torch.int32)  # Destination tote IDs for each environment
                env_indices = torch.arange(env.num_envs, device=env.unwrapped.device)  # Indices of environments

                # Get the objects that can be packed
                packable_objects, packable_mask = get_packable_object_indices(
                    num_obj_per_env, tote_manager, env_indices, tote_ids
                )

                # Select random packable objects
                new_obj_idx = select_random_packable_objects(
                    packable_objects, packable_mask, env.unwrapped.device, num_obj_per_env
                )

                # Increment packing counters for environments with valid object indices
                packing_counters += (new_obj_idx > 0).int()
                obj_idx = new_obj_idx

                actions[:, 1:] = torch.cat(
                    [
                        obj_idx.unsqueeze(1),
                        torch.tensor([0, 0, 0, 0, 0, 1, 0], device=env.unwrapped.device).repeat(batch_size, 1),
                    ],
                    dim=1,
                )

                # Vectorized height calculation
                height_increment = 0.05
                drop_heights = height_increment * packing_counters
                # Only apply drop height for non-zero object indices
                drop_heights = drop_heights * (obj_idx > 0).float()

                actions[:, 4] = drop_heights  # Set z-drop height based on packing order

                # print(f"[INFO]: Actions: {actions}")
            # apply actions
            env.step(actions)

            if torch.all(obj_idx == 0):
                # Wait to show that it has been packed before resetting
                for i in range(50):
                    env.step(torch.zeros(env.action_space.shape, device=env.unwrapped.device))
                obj_idx = torch.ones(env.num_envs, device=env.unwrapped.device, dtype=torch.int32)
                # Reset packing counters when environment is reset
                packing_counters = torch.zeros(env.num_envs, device=env.unwrapped.device, dtype=torch.int32)
                env.reset()

            idx += 1

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
