# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import multiprocessing as mp
import time

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import torch
from isaaclab.envs import DirectRLEnv, ManagerBasedRLEnv
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from packing3d import Attitude, Position, Transform
from tote_consolidation.tasks.manager_based.pack.utils.tote_helpers import (
    calculate_rotated_bounding_box,
)


def _process_single_item(args):
    """Process a single item for multiprocessing.

    Args:
        args: Tuple containing (i, actions_i, x_i, y_i, qz_i, qw_i, rotated_half_dim_i, bpp_problem_i)

    Returns:
        Tuple of (i, z_value, packed_obj_idx)
    """
    i, actions_i, x_i, y_i, qz_i, qw_i, rotated_half_dim_i, bpp_problem_i = args

    # Analytically determine the z position of the object
    curr_attitude = Attitude(0, 0, 0) if qz_i == 0 and qw_i == 1 else Attitude(0, 0, torch.pi / 2)
    item_to_add = bpp_problem_i.items[int(actions_i[1])]
    item_to_add.rotate(curr_attitude)
    item_to_add.calc_heightmap()

    # Convert grid indices to actual coordinates
    x_coord = int(np.floor(x_i).item())
    y_coord = int(np.floor(y_i).item())
    bpp_problem_i.container.add_item_topdown(item_to_add, x_coord, y_coord)

    transform = Transform(
        Position(bpp_problem_i.container.geometry.x_size - x_coord,
                bpp_problem_i.container.geometry.y_size - y_coord,
                item_to_add.position.z),
        curr_attitude,
    )
    item_to_add.transform(transform)
    bpp_problem_i.container.add_item(item_to_add)

    z_value = rotated_half_dim_i[2] + item_to_add.position.z / 100
    packed_obj_idx = actions_i[1]

    return i, z_value, packed_obj_idx

class RslRlGCUVecEnvWrapper(RslRlVecEnvWrapper):
    """
    Inherit the RSL-RL wrapper library for GCU Lab environments.
    This specifies the tote id and the object id in the action tensor,
    which is part of a PackingAction, but not in the action space of the policy.
    """

    def __init__(self, env: ManagerBasedRLEnv | DirectRLEnv, clip_actions: float | None = None):
        """Initializes the wrapper.

        Note:
            The wrapper calls :meth:`reset` at the start since the RSL-RL runner does not call reset.

        Args:
            env: The environment to wrap around.
            clip_actions: The clipping value for actions. If ``None``, then no clipping is done.

        Raises:
            ValueError: When the environment is not an instance of :class:`ManagerBasedRLEnv` or :class:`DirectRLEnv`.
        """
        super().__init__(env, clip_actions)

        # Initialize multiprocessing pool
        self.mp_pool = None
        self.mp_enabled = False  # Can be toggled for debugging

    def _convert_to_pos_quat(self, actions: torch.Tensor) -> torch.Tensor:
        orientation_idx = actions[:, 2]
        theta_rad = orientation_idx * (torch.pi / 2)  # 0 or pi/2 radians (0 or 90 degrees)
        qx = torch.sin(theta_rad / 2)
        qy = torch.zeros_like(theta_rad)
        qz = torch.zeros_like(theta_rad)
        qw = torch.cos(theta_rad / 2)

        # Convert (x, y, theta) actions to (x, y, z, qx, qy, qz, qw)
        # Assume z=0 for placement (to be updated), and theta is in radians
        x = actions[:, 0]
        y = actions[:, 1]


        bbox_offset = self.env.unwrapped.tote_manager.obj_bboxes[
            torch.arange(actions.shape[0], device=self.env.unwrapped.device), torch.zeros(actions.shape[0], device=self.env.unwrapped.device).int()  # TODO (kaikwan): fix this hardcoded index to check with selected object
        ]
        quats = torch.stack([qx, qy, qz, qw], dim=1)  # shape [batch, 4]
        rotated_dim = (
            calculate_rotated_bounding_box(
                bbox_offset, quats, device=self.env.unwrapped.device
            )
        )
        x_pos_range = self.env.unwrapped.tote_manager.true_tote_dim[0] / 100 - rotated_dim[:, 0]
        y_pos_range = self.env.unwrapped.tote_manager.true_tote_dim[1] / 100 - rotated_dim[:, 1]
        x = torch.sigmoid(x) * (self.env.unwrapped.tote_manager.true_tote_dim[0] / 100 - rotated_dim[:, 0])
        y = torch.sigmoid(y) * (self.env.unwrapped.tote_manager.true_tote_dim[1] / 100 - rotated_dim[:, 1])

        # Compute z analytically for each sample in the batch using multiprocessing
        z = torch.zeros_like(x)

        return torch.stack([x, y, z, quats[:, 0], quats[:, 1], quats[:, 2], quats[:, 3]], dim=1), [x_pos_range, y_pos_range], rotated_dim

    def _get_z_position_from_depth(self, image_obs: torch.Tensor, xy_pos: torch.Tensor, xy_pos_range: torch.Tensor, rotated_dim: torch.Tensor) -> torch.Tensor:
        """Get the z position from the depth image."""
        depth_img = image_obs.reshape(self.env.unwrapped.num_envs, 37, 52) # TODO (kaikwan): get shape smarter

        # Rescale x_pos and y_pos to the range of the depth image
        total_tote_x = self.env.unwrapped.tote_manager.true_tote_dim[0] / 100
        total_tote_y = self.env.unwrapped.tote_manager.true_tote_dim[1] / 100
        x_pos = torch.round(52 -(xy_pos[0] / total_tote_x) * 52).to(torch.int64)
        y_pos = torch.round((xy_pos[1] / total_tote_y) * 37).to(torch.int64)

        # Compute patch extents in pixel units by scaling world dimensions to pixel coordinates
        # The image covers the total tote dimensions, so scale object dimensions relative to total tote dimensions
        x_extent = torch.round((rotated_dim[:, 0] / total_tote_x) * 52).clamp(min=1).long()
        y_extent = torch.round((rotated_dim[:, 1] / total_tote_y) * 37).clamp(min=1).long()

        # Compute patch start/end indices, clamp to image bounds
        x1 = x_pos.clamp(0, 51)
        y0 = y_pos.clamp(0, 36)
        x0 = (x1 - x_extent).clamp(0, 51)
        y1 = (y0 + y_extent).clamp(0, 36)

        # Create batch indices for advanced indexing
        batch_idx = torch.arange(depth_img.shape[0], device=self.device)

        # For each sample, extract the patch and get the max value
        # Use broadcasting to build masks for all pixels in one go
        img_h, img_w = 37, 52
        grid_y = torch.arange(img_h, device=self.device).view(1, img_h, 1)
        grid_x = torch.arange(img_w, device=self.device).view(1, 1, img_w)
        y0_ = y0.view(-1, 1, 1)
        y1_ = y1.view(-1, 1, 1)
        x0_ = x0.view(-1, 1, 1)
        x1_ = x1.view(-1, 1, 1)
        mask = (grid_y >= y0_) & (grid_y <= y1_) & (grid_x >= x0_) & (grid_x <= x1_)

        # Masked min: set out-of-patch values and zeros to inf, then take min
        depth_img_masked = depth_img.clone()
        depth_img_masked[~mask] = float('inf')
        depth_img_masked[depth_img_masked == 0] = float('inf')
        z_pos = depth_img_masked.view(depth_img.shape[0], -1).min(dim=1).values

        z_pos = 20.01 - z_pos
        z_pos = z_pos.clamp(min=0.0, max=0.4)
        import matplotlib.pyplot as plt
        plt.imshow(depth_img[0].cpu().numpy(), cmap='viridis')
        plt.colorbar()
        plt.savefig("depth_image.png")
        plt.close()
        # Plot the depth image with the max_z_pos
        plt.imshow(depth_img_masked[0].cpu().numpy(), cmap='viridis')
        plt.colorbar()
        plt.savefig("depth_image_with_max_z_pos.png")
        plt.close()
        return z_pos

    def step(self, actions: torch.Tensor, image_obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:

        # Get z_pos from depth image
        actions, xy_pos_range, rotated_dim = self._convert_to_pos_quat(actions)
        z_pos = self._get_z_position_from_depth(image_obs, [actions[:, 0], actions[:, 1]], xy_pos_range, rotated_dim)

        tote_ids = torch.zeros(self.env.unwrapped.num_envs, device=self.env.unwrapped.device).int()
        packable_objects = self.env.unwrapped.bpp.get_packable_object_indices(self.env.unwrapped.tote_manager.num_objects, self.env.unwrapped.tote_manager, torch.arange(self.env.unwrapped.num_envs, device=self.env.unwrapped.device), tote_ids)[0]
        actions = torch.cat(
            [
                tote_ids.unsqueeze(1).to(self.env.unwrapped.device),  # Destination tote IDs
                torch.tensor([row[0] for row in packable_objects], device=self.env.unwrapped.device).unsqueeze(1),  # Object indices
                actions,
            ], dim=1
        )
        actions[:, 4] = z_pos

        # clip actions
        if self.clip_actions is not None:
            actions = torch.clamp(actions, -self.clip_actions, self.clip_actions)
        # record step information
        obs_dict, rew, terminated, truncated, extras = self.env.unwrapped.step(actions)

        # compute dones for compatibility with RSL-RL
        dones = (terminated | truncated).to(dtype=torch.long)
        # move extra observations to the extras dict
        obs = obs_dict["policy"]
        extras["observations"] = obs_dict
        # move time out information to the extras dict
        # this is only needed for infinite horizon tasks
        if not self.unwrapped.cfg.is_finite_horizon:
            extras["time_outs"] = truncated

        # return the step information
        return obs, rew, dones, extras
