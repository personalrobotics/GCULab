# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

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
import tote_consolidation.tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg
import matplotlib.pyplot as plt
import isaaclab.utils.math as math_utils
from typing import Tuple
import numpy as np

def depth_to_heightmap(depth_img, K, cam_T_world, grid_res=0.01, x_crop_bounds=None, y_crop_bounds=None, z_crop_bounds=None):
    """
    depth_img: (H, W) perspective depth in meters (distance along camera ray)
    K: (3, 3) intrinsics
    cam_T_world: (4, 4) camera-to-world matrix  [R|t] that maps camera coords -> world coords
    grid_res: meters per pixel in the output heightmap
    x_crop_bounds: tuple (xmin, xmax) to crop the world X coordinates
    y_crop_bounds: tuple (ymin, ymax) to crop the world Y coordinates
    z_crop_bounds: tuple (zmin, zmax) to crop the world Z coordinates
    """
    H, W = depth_img.shape
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    # 1) Build per-pixel ray in camera coords: r = K^{-1} [u, v, 1]^T (not unit yet)
    u, v = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))
    x = (u - cx) / fx
    y = (v - cy) / fy
    # Ray direction (not normalized): [x, y, 1]
    # For perspective depth d_ray (distance along the ray),
    # the 3D point in camera coords is:
    #    P_cam = (d_ray / ||[x, y, 1]||) * [x, y, 1]
    # This is the key fix vs. your code.
    ray_norm = np.sqrt(x*x + y*y + 1.0)
    d_ray = depth_img.astype(np.float32)
    scale = (d_ray / ray_norm)  # (H, W)

    Xc = x * scale
    Yc = y * scale
    Zc = 1.0 * scale

    # 2) Transform to world coords
    Pc = np.stack([Xc, Yc, Zc, np.ones_like(Zc)], axis=-1).reshape(-1, 4)  # (H*W, 4)
    Pw = (Pc @ cam_T_world.T)[:, :3]  # (H*W, 3)
    xw, yw, zw = Pw[:, 0], Pw[:, 1], Pw[:, 2]

    # Print min and max points in world coord of each axis
    print(f"World X-axis: min={xw.min():.3f}, max={xw.max():.3f}")
    print(f"World Y-axis: min={yw.min():.3f}, max={yw.max():.3f}")
    print(f"World Z-axis: min={zw.min():.3f}, max={zw.max():.3f}")

    # Apply crop bounds if provided
    if x_crop_bounds is not None:
        mask = (xw >= x_crop_bounds[0]) & (xw <= x_crop_bounds[1])
        xw, yw, zw = xw[mask], yw[mask], zw[mask]
    
    if y_crop_bounds is not None:
        mask = (yw >= y_crop_bounds[0]) & (yw <= y_crop_bounds[1])
        xw, yw, zw = xw[mask], yw[mask], zw[mask]

    if z_crop_bounds is not None:
        mask = (zw >= z_crop_bounds[0]) & (zw <= z_crop_bounds[1])
        xw, yw, zw = xw[mask], yw[mask], zw[mask]

    # 3) Bin into a top-down XY grid, keeping min world-Z per cell
    xmin, xmax = float(xw.min()), float(xw.max())
    ymin, ymax = float(yw.min()), float(yw.max())
    # Avoid zero-size bounds
    if xmax - xmin < 1e-6: xmax += 1e-3
    if ymax - ymin < 1e-6: ymax += 1e-3

    GW = int(np.ceil((xmax - xmin) / grid_res))
    GH = int(np.ceil((ymax - ymin) / grid_res))
    heightmap = np.full((GW, GH), np.nan, dtype=np.float32)

    gx = np.clip(((xw - xmin) / grid_res).astype(np.int32), 0, GW - 1)
    gy = np.clip(((yw - ymin) / grid_res).astype(np.int32), 0, GH - 1)
    # Flip gx to correct front-to-back orientation
    gx = GW - 1 - gx
    # vectorized "min per cell"
    flat_idx = gx * GH + gy
    # Keep lowest world-Z per cell
    # Use argsort to reduce then take first occurrence per index
    order = np.argsort(flat_idx, kind="mergesort")
    flat_idx_sorted = flat_idx[order]
    zw_sorted = zw[order]

    # find segment ends (last index for each unique cell)
    uniq, last_idx = np.unique(flat_idx_sorted, return_index=False, return_counts=True)
    last_idx = np.cumsum(last_idx) - 1  # last occurrence positions
    # min within each segment
    # we can do a segmented min by scanning; for simplicity, use numpy grouping via split
    # but a faster way:
    min_per_cell = np.full_like(uniq, np.inf, dtype=np.float32)
    start = 0
    for i, end in enumerate(last_idx):
        seg = zw_sorted[start:end+1]
        min_per_cell[i] = np.min(seg)
        start = end + 1

    # write back - convert linear indices back to 2D coordinates for proper assignment
    gx_uniq = uniq // GH
    gy_uniq = uniq % GH
    heightmap[gx_uniq, gy_uniq] = min_per_cell

    return heightmap


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
    # simulate environment
    while simulation_app.is_running():
        # run everything in inference mode
        with torch.inference_mode():
            # compute zero actions
            actions = torch.zeros(env.action_space.shape, device=env.unwrapped.device)
            # apply actions
            obs, reward, terminated, truncated, info = env.step(actions)
            plt.imshow(obs["sensor"][0, :, :, 0].cpu().numpy())
            plt.savefig("depth.png")
            plt.imshow(obs["sensor"][0, :, :, 1:4].cpu().numpy())
            plt.savefig("sensor.png")
            # depth_image = obs["sensor"][:, :, :, 0]
            # extrinsics_pos_w = env.unwrapped.scene.sensors['tiled_camera'].data.pos_w # (num_envs, 3)
            # extrinsics_pos_w = extrinsics_pos_w - env.unwrapped.scene.env_origins # (num_envs, 3)
            # extrinsics_quat_w_world = torch.tensor([-0.69035, -0.15305, -0.15305, -0.69035], device=env.unwrapped.device).repeat(env.unwrapped.num_envs, 1) # (num_envs, 4)
            # # Create extrinsics matrix from position and quaternion
            # extrinsics_mat_w = torch.eye(4, device=env.unwrapped.device).repeat(env.unwrapped.num_envs, 1, 1)
            # extrinsics_mat_w[:, :3, :3] = math_utils.matrix_from_quat(extrinsics_quat_w_world)
            # extrinsics_mat_w[:, :3, 3] = extrinsics_pos_w
            # heightmap = depth_to_heightmap(depth_image[0].cpu().numpy().squeeze(), env.unwrapped.scene.sensors['tiled_camera'].data.intrinsic_matrices[0].cpu().numpy(), extrinsics_mat_w[0].cpu().numpy(), 
            # grid_res=0.01,
            # x_crop_bounds=(0.14, 0.8), 
            # y_crop_bounds=(-1.0, 0.2),
            # z_crop_bounds=(1.4, 1.94))
            # plt.clf()
            # plt.imshow(heightmap)
            # plt.savefig("heightmap.png")

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
