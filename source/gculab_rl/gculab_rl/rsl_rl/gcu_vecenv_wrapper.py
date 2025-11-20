# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import multiprocessing as mp

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import torch
from isaaclab.envs import DirectRLEnv, ManagerBasedRLEnv
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from packing3d import Attitude, Position, Transform
from geodude.tasks.manager_based.pack.utils.tote_helpers import (
    calculate_rotated_bounding_box,
)

PI = 3.141592653589793238462643383279502884


def make_quaternion_prototypes(device="cpu", n_side=1):
    """
    Generate quaternions uniformly on SO(3) using Hopf coordinates
    and HEALPix grids

    Outputs:
        prototypes: generated quaternion prototypes (n_grid, 4)
        where n_grid == 12*n_side**3*sqrt(12*PI)
    """
    # Uniformly sample the 3D sphere
    n_pix = 12 * n_side**2
    p = torch.arange(12 * n_side**2, device=device)
    ph = (p + 1) / 2
    i1 = torch.floor(torch.sqrt(ph - torch.sqrt(torch.floor(ph)))) + 1
    j1 = p + 1 - 2 * i1 * (i1 - 1)
    valid = torch.logical_and(i1 < n_side, j1 <= 4 * i1)
    i1 = i1[valid]
    j1 = j1[valid]
    z1 = 1 - i1**2 / 3 / n_side**2
    s = 1
    phi1 = PI / 2 / i1 * (j1 - s / 2)
    theta1 = torch.acos(z1)

    ph = p - 2 * n_side * (n_side - 1)
    i2 = torch.floor(ph / 4 / n_side) + n_side
    j2 = ph % (4 * n_side) + 1
    j0 = j2[i2 == 2 * n_side]
    s = (n_side + 1) % 2
    phi0 = PI / 2 / n_side * (j0 - s / 2)
    theta0 = PI / 2 * torch.ones_like(phi0)
    valid = torch.logical_and(n_side <= i2, i2 < 2 * n_side)
    i2 = i2[valid]
    j2 = j2[valid]
    z2 = 4 / 3 - 2 * i2 / 3 / n_side
    s = (i2 - n_side + 1) % 2
    phi2 = PI / 2 / n_side * (j2 - s / 2)
    theta2 = torch.acos(z2)

    theta = torch.concat([theta1, PI - theta1, theta2, PI - theta2, theta0], dim=0).reshape(1, -1)
    phi = torch.concat([phi1, phi1, phi2, phi2, phi0], dim=0).reshape(1, -1)

    # Generate prototypes using Hopf fibration
    n1 = torch.floor(torch.sqrt(torch.tensor(PI * n_pix, device=device)))
    psi = torch.arange(0, 2 * PI, step=2 * PI / n1, device=device).reshape(-1, 1)
    x1 = torch.cos(theta / 2) * torch.cos(psi / 2)
    x2 = torch.sin(theta / 2) * torch.sin(phi - psi / 2)
    x3 = torch.sin(theta / 2) * torch.cos(phi - psi / 2)
    x4 = torch.cos(theta / 2) * torch.sin(psi / 2)

    prototypes = torch.stack([x1, x2, x3, x4], dim=-1).reshape(-1, 4)
    prototypes = prototypes / torch.norm(prototypes, dim=-1, keepdim=True)
    return prototypes


def quaternion_from_bin_logits(logits):
    """
    Reconstruct coarse quaternions from bin logits
    Inputs:
      logits: logits of quaternion belonging to each bin (N, M)

    Outputs:
      quaternions: reconstructed quaternions (N, 4)
    """
    batch_size, num_bins = logits.shape
    prototypes = make_quaternion_prototypes(device=logits.device)
    assert prototypes.shape[0] == num_bins, f"Prototypes shape: {prototypes.shape}, num_bins: {num_bins}"
    prototypes_rep = torch.repeat_interleave(prototypes.unsqueeze(0), batch_size, dim=0)
    indices = torch.repeat_interleave(torch.argmax(logits, dim=-1, keepdim=True), repeats=4, dim=1)
    quaternions = torch.gather(prototypes_rep, dim=1, index=indices.unsqueeze(1))
    return torch.squeeze(quaternions, dim=1)


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

    def _convert_to_pos_quat(self, actions: torch.Tensor, object_to_pack: list) -> torch.Tensor:
        orientation_logits = actions[:, 2:]  # shape [batch, 4]
        assert orientation_logits.shape[1] == 72, f"Orientation logits shape: {orientation_logits.shape}"

        # Convert orientation logits to quaternions using bin logits
        quats = quaternion_from_bin_logits(orientation_logits)

        # Convert (x, y, theta) actions to (x, y, z, qx, qy, qz, qw)
        # Assume z=0 for placement (to be updated), and theta is in radians
        x = actions[:, 0]
        y = actions[:, 1]

        bbox_offset = torch.stack([
            self.env.unwrapped.tote_manager.get_object_bbox(env_idx, obj_idx)
            for env_idx, obj_idx in zip(
                torch.arange(actions.shape[0], device=self.env.unwrapped.device),
                object_to_pack,
            )
        ])
        
        rotated_dim = calculate_rotated_bounding_box(bbox_offset, quats, device=self.env.unwrapped.device)
        x_pos_range = self.env.unwrapped.tote_manager.true_tote_dim[0] / 100 - rotated_dim[:, 0]
        y_pos_range = self.env.unwrapped.tote_manager.true_tote_dim[1] / 100 - rotated_dim[:, 1]
        x = torch.sigmoid(5 * x) * (self.env.unwrapped.tote_manager.true_tote_dim[0] / 100 - rotated_dim[:, 0])
        y = torch.sigmoid(5 * y) * (self.env.unwrapped.tote_manager.true_tote_dim[1] / 100 - rotated_dim[:, 1])

        # Compute z analytically for each sample in the batch using multiprocessing
        z = torch.zeros_like(x)

        return (
            torch.stack([x, y, z, quats[:, 0], quats[:, 1], quats[:, 2], quats[:, 3]], dim=1),
            [x_pos_range, y_pos_range],
            rotated_dim,
        )

    def _get_z_position_from_depth(
        self, image_obs: torch.Tensor, xy_pos: torch.Tensor, xy_pos_range: torch.Tensor, rotated_dim: torch.Tensor
    ) -> torch.Tensor:
        """Get the z position from the depth image."""
        img_h = self.env.unwrapped.observation_space["sensor"].shape[-3]
        img_w = self.env.unwrapped.observation_space["sensor"].shape[-2]
        depth_img = image_obs.reshape(self.env.unwrapped.num_envs, img_h, img_w)

        # Rescale x_pos and y_pos to the range of the depth image
        total_tote_x = self.env.unwrapped.tote_manager.true_tote_dim[0] / 100
        total_tote_y = self.env.unwrapped.tote_manager.true_tote_dim[1] / 100
        tote_x_m = self.env.unwrapped.tote_manager.true_tote_dim[0]
        tote_y_m = self.env.unwrapped.tote_manager.true_tote_dim[1]
        x_pos = torch.round(tote_x_m - (xy_pos[0] / total_tote_x) * tote_x_m).to(torch.int64)
        y_pos = torch.round((xy_pos[1] / total_tote_y) * tote_y_m).to(torch.int64)

        # Compute patch extents in pixel units by scaling world dimensions to pixel coordinates
        # The image covers the total tote dimensions, so scale object dimensions relative to total tote dimensions
        x_extent = torch.round((rotated_dim[:, 0] / total_tote_x) * tote_x_m).clamp(min=1).long()
        y_extent = torch.round((rotated_dim[:, 1] / total_tote_y) * tote_y_m).clamp(min=1).long()

        # Compute patch start/end indices, clamp to image bounds
        x1 = x_pos.clamp(0, tote_x_m)
        y0 = y_pos.clamp(0, tote_y_m)
        x0 = (x1 - x_extent).clamp(0, tote_x_m)
        y1 = (y0 + y_extent).clamp(0, tote_y_m)

        # For each sample, extract the patch and get the max value
        # Use broadcasting to build masks for all pixels in one go
        grid_y = torch.arange(img_h, device=self.device).view(1, img_h, 1)
        grid_x = torch.arange(img_w, device=self.device).view(1, 1, img_w)
        y0_ = y0.view(-1, 1, 1)
        y1_ = y1.view(-1, 1, 1)
        x0_ = x0.view(-1, 1, 1)
        x1_ = x1.view(-1, 1, 1)
        mask = (grid_y >= y0_) & (grid_y <= y1_) & (grid_x >= x0_) & (grid_x <= x1_)

        # Masked min: set out-of-patch values and zeros to inf, then take min
        depth_img_masked = depth_img.clone()
        depth_img_masked[~mask] = float("inf")
        depth_img_masked[depth_img_masked == 0] = float("inf")
        z_pos = depth_img_masked.view(depth_img.shape[0], -1).min(dim=1).values

        z_pos = 20.0 - z_pos
        z_pos = z_pos.clamp(min=0.0, max=0.4)

        return z_pos

    def step(
        self, actions: torch.Tensor, image_obs: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        tote_ids = torch.zeros(self.env.unwrapped.num_envs, device=self.env.unwrapped.device).int()
        packable_objects = self.env.unwrapped.bpp.get_packable_object_indices(
            self.env.unwrapped.tote_manager.num_objects,
            self.env.unwrapped.tote_manager,
            torch.arange(self.env.unwrapped.num_envs, device=self.env.unwrapped.device),
            tote_ids,
        )[0]
        object_to_pack = [row[0] for row in packable_objects]
        for i in range(self.env.unwrapped.num_envs):
            self.unwrapped.bpp.packed_obj_idx[i].append(torch.tensor([object_to_pack[i].item()]))

        actions, xy_pos_range, rotated_dim = self._convert_to_pos_quat(actions, object_to_pack)

        # Get z_pos from depth image
        z_pos = self._get_z_position_from_depth(image_obs, [actions[:, 0], actions[:, 1]], xy_pos_range, rotated_dim)
        actions = torch.cat(
            [
                tote_ids.unsqueeze(1).to(self.env.unwrapped.device),  # Destination tote IDs
                torch.tensor(object_to_pack, device=self.env.unwrapped.device).unsqueeze(1),  # Object indices
                actions,
            ],
            dim=1,
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
