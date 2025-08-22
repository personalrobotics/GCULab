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
from tote_consolidation.tasks.manager_based.pack.utils.tote_helpers import (
    calculate_rotated_bounding_box,
)
import isaaclab.utils.math as math_utils


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

    # batch*n
    def normalize_vector(self, v):
        batch=v.shape[0]
        v_mag = torch.sqrt(v.pow(2).sum(1))# batch
        v_mag = torch.max(v_mag, torch.autograd.Variable(torch.FloatTensor([1e-8]).cuda()))
        v_mag = v_mag.view(batch,1).expand(batch,v.shape[1])
        v = v/v_mag
        return v
        
    # u, v batch*n
    def cross_product(self, u, v):
        batch = u.shape[0]
        #print (u.shape)
        #print (v.shape)
        i = u[:,1]*v[:,2] - u[:,2]*v[:,1]
        j = u[:,2]*v[:,0] - u[:,0]*v[:,2]
        k = u[:,0]*v[:,1] - u[:,1]*v[:,0]
            
        out = torch.cat((i.view(batch,1), j.view(batch,1), k.view(batch,1)),1)#batch*3
            
        return out
    
    # batch*6
    def compute_rotation_matrix_from_ortho6d(self, poses):
        x_raw = poses[:,0:3]#batch*3
        y_raw = poses[:,3:6]#batch*3
            
        x = self.normalize_vector(x_raw) #batch*3
        z = self.cross_product(x,y_raw) #batch*3
        z = self.normalize_vector(z)#batch*3
        y = self.cross_product(z,x)#batch*3
            
        x = x.view(-1,3,1)
        y = y.view(-1,3,1)
        z = z.view(-1,3,1)
        matrix = torch.cat((x,y,z), 2) #batch*3*3
        return matrix


    def _convert_6d_rotation_to_quaternion(self, rotation_6d: torch.Tensor) -> torch.Tensor:
        """Convert 6D rotation representation to quaternion.
        
        Args:
            rotation_6d: 6D rotation tensor [batch, 6]
        Returns:
            Quaternion tensor [batch, 4]
        """
        r_matrix = self.compute_rotation_matrix_from_ortho6d(rotation_6d)
        # Convert rotation matrix to quaternion
        quats = math_utils.quat_from_matrix(r_matrix)  # batch*4
        
        return quats

    def _convert_to_pos_quat(self, actions: torch.Tensor, object_to_pack: list) -> torch.Tensor:
        quat_pred = actions[:, 2:]  # shape [batch, 6]
        assert quat_pred.shape[1] == 6, "Expected quaternion to have 6 components for 6D rotation representation, but got {}".format(quat_pred.shape[1])

        # Convert 6D rotation representation to 4D quaternion using helper function
        quats = self._convert_6d_rotation_to_quaternion(quat_pred)

        # Convert (x, y, theta) actions to (x, y, z, qx, qy, qz, qw)
        # Assume z=0 for placement (to be updated), and theta is in radians
        x = actions[:, 0]
        y = actions[:, 1]


        bbox_offset = self.env.unwrapped.tote_manager.obj_bboxes[
            torch.arange(actions.shape[0], device=self.env.unwrapped.device), torch.tensor(object_to_pack, device=self.env.unwrapped.device)
        ]
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
        img_h = self.env.unwrapped.observation_space['sensor'].shape[-3]
        img_w = self.env.unwrapped.observation_space['sensor'].shape[-2]
        depth_img = image_obs.reshape(self.env.unwrapped.num_envs, img_h, img_w) # TODO (kaikwan): get shape smarter

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
        depth_img_masked[~mask] = float('inf')
        depth_img_masked[depth_img_masked == 0] = float('inf')
        z_pos = depth_img_masked.view(depth_img.shape[0], -1).min(dim=1).values

        z_pos = 20.01 - z_pos
        z_pos = z_pos.clamp(min=0.0, max=0.4)

        return z_pos

    def step(self, actions: torch.Tensor, image_obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        tote_ids = torch.zeros(self.env.unwrapped.num_envs, device=self.env.unwrapped.device).int()
        packable_objects = self.env.unwrapped.bpp.get_packable_object_indices(self.env.unwrapped.tote_manager.num_objects, self.env.unwrapped.tote_manager, torch.arange(self.env.unwrapped.num_envs, device=self.env.unwrapped.device), tote_ids)[0]
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
