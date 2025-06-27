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
parser.add_argument("--discretize_factor", type=float, default=5, help="Discretization factor.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import torch

try:
    from pyconsolidation.tetris_beam_search_mc_torch import (
        get_place_poses_from_placements,
        monte_carlo_beam_search,
        visualize_board_voxels,
    )
except ImportError:
    print("[ERROR]: pyconsolidation module not found. Please install it'.")
    raise


import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg

# PLACEHOLDER: Extension template (do not remove this comment)


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

    num_bins = [
        env.unwrapped.gcu.planning_tote_dim[0] // args_cli.discretize_factor,
        env.unwrapped.gcu.planning_tote_dim[1] // args_cli.discretize_factor,
        env.unwrapped.gcu.planning_tote_dim[2] // args_cli.discretize_factor,
    ]

    blocks = env.unwrapped.gcu.obj_bboxes
    # divide by discretize_factor to get the number of bins
    blocks = torch.ceil(torch.tensor(blocks) / args_cli.discretize_factor).to(dtype=torch.int)

    place_poses = None

    for block in blocks:
        import time

        start = time.time()
        board_and_seq = monte_carlo_beam_search(
            block,
            W=int(num_bins[0]),
            H=int(num_bins[2]),
            D=int(num_bins[1]),
            allow_rotations=True,
            beam_width=3,
            sample_size=3,
            trials=1,
        )
        print(f"[INFO]: monte_carlo_beam_search time: {time.time() - start}")
        start = time.time()
        board, sequence, id_to_index = board_and_seq
        visualize_board_voxels(board)
        place_poses = get_place_poses_from_placements(sequence, id_to_index, block, flip=True)

    def compute_place_pos_tote_centered(obj_pos, obj_rotated_dims, planning_tote_dim):
        """
        Compute the position of the object in the tote's coordinate system.
        obj_pos: The 3D position of the object in the tote coordinate,
                 where the origin is at the bottom left corner of the tote.
        obj_rotated_dims: The dimensions of the object after it has been rotated,
                           represented as a 3D tensor (length, width, height).
        planning_tote_dim: The dimensions of the tote in the world coordinate system,
                  represented as a 3D tensor (length, width, height).
        """
        obj_pos = obj_pos.float() + obj_rotated_dims / 2  # Shift the object to its center
        obj_pos_tote = -obj_pos

        place_pos_tote = torch.zeros(3, device=obj_pos_tote.device)
        place_pos_tote[0] = obj_pos_tote[0] + planning_tote_dim[0] / 2
        place_pos_tote[1] = obj_pos_tote[1] + planning_tote_dim[1] / 2
        place_pos_tote[2] = torch.abs(obj_pos_tote[2])

        return place_pos_tote

    idx = 0
    seq_idx = 0
    place_step_interval = 100  # time interval to place objects
    # simulate environment
    while simulation_app.is_running():
        # run everything in inference mode
        with torch.inference_mode():
            # compute zero actions
            actions = torch.zeros(env.action_space.shape, device=env.unwrapped.device)
            if idx % place_step_interval == 0 and idx != 0:
                obj_info = place_poses[seq_idx]
                obj_idx = obj_info["block_index"]
                place_rot_tote = torch.tensor(obj_info["rotation_quat"], device=env.unwrapped.device)

                obj_pos = torch.tensor(obj_info["position"], device=env.unwrapped.device) * 5
                obj_rotated_dims = torch.tensor(obj_info["rotated_dims"], device=env.unwrapped.device) * 5

                place_pos_tote = (
                    compute_place_pos_tote_centered(obj_pos, obj_rotated_dims, env.unwrapped.gcu.planning_tote_dim)
                    / 100.0
                )  # convert to meters

                # TODO (kaikwan): clamp the tote position to the tote dimensions
                #   No need currently since bbox is an overestimate

                place_pos_tote = torch.clamp(
                    place_pos_tote,
                    min=place_pos_tote - obj_rotated_dims / 200,
                    max=place_pos_tote + obj_rotated_dims / 200,
                )
                # Hack to make place pose lower
                place_pos_tote[2] = place_pos_tote[2] * 0.8
                actions[:, 0] = obj_idx + 1
                actions[:, 1:4] = torch.tensor(place_pos_tote, device=env.unwrapped.device)
                actions[:, 4:8] = torch.tensor(place_rot_tote, device=env.unwrapped.device)
                seq_idx += 1
                print(f"[INFO]: Actions: {actions}")
            # apply actions
            env.step(actions)
            idx += 1

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
