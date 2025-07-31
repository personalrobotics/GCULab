# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import multiprocessing as mp
import gymnasium as gym
import time
import torch
import numpy as np


from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from isaaclab.envs import ManagerBasedRLEnv, DirectRLEnv
from tote_consolidation.tasks.manager_based.pack.utils.tote_helpers import (
    calculate_rotated_bounding_box,
)
from packing3d import (Attitude, Transform, Position)
import matplotlib.pyplot as plt

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
        """Convert actions to position and quaternion.

        Args:
            actions: The actions to convert. (x, y, orientation)

        Returns:
            The position and quaternion. (x, y, z, qx, qy, qz, qw)
        """
        start_time = time.time()
        orientation_idx = actions[:, 5]
        theta_rad = orientation_idx * (torch.pi / 2)  # 0 or pi/2 radians (0 or 90 degrees)
        qx = torch.sin(theta_rad / 2)
        qy = torch.zeros_like(theta_rad)
        qz = torch.zeros_like(theta_rad)
        qw = torch.cos(theta_rad / 2)

        # Convert (x, y, theta) actions to (x, y, z, qx, qy, qz, qw)
        # Assume z=0 for placement, and theta is in radians
        x = actions[:, 2]
        y = actions[:, 3]
        bbox_offset = self.env.unwrapped.tote_manager.obj_bboxes[
            torch.arange(actions.shape[0], device=self.env.unwrapped.device), actions[:, 1].long()
        ]
        quats = torch.stack([qx, qy, qz, qw], dim=1)  # shape [batch, 4]
        rotated_half_dim = (
            calculate_rotated_bounding_box(
                bbox_offset, quats, device=self.env.unwrapped.device
            )
            / 2.0
        )
        x = torch.sigmoid(x) * (self.env.unwrapped.tote_manager.true_tote_dim[0] / 100 - 2 * rotated_half_dim[:, 0])
        y = torch.sigmoid(y) * (self.env.unwrapped.tote_manager.true_tote_dim[1] / 100 - 2 * rotated_half_dim[:, 1])

        # Compute z analytically for each sample in the batch using multiprocessing
        z = torch.zeros_like(x)
        
        if self.mp_enabled and actions.shape[0] > 1:  # Only use multiprocessing for multiple items
            try:
                # Initialize pool if not already done
                if self.mp_pool is None:
                    self.mp_pool = mp.Pool(processes=min(mp.cpu_count(), actions.shape[0]))
                
                # Prepare data for multiprocessing
                mp_args = []
                for i in range(actions.shape[0]):
                    # Create a deep copy of the BPP problem to avoid conflicts
                    bpp_problem_copy = self.env.unwrapped.bpp.problems[i]
                    mp_args.append((
                        i, 
                        actions[i].cpu().numpy(),  # Convert to numpy for multiprocessing
                        x[i].cpu().numpy(),
                        y[i].cpu().numpy(),
                        qz[i].cpu().numpy(),
                        qw[i].cpu().numpy(),
                        rotated_half_dim[i].cpu().numpy(),
                        bpp_problem_copy
                    ))
                
                # Process in parallel
                results = self.mp_pool.map(_process_single_item, mp_args)
                
                # Collect results
                for i, z_value, packed_obj_idx in results:
                    z[i] = torch.tensor(z_value, device=self.env.unwrapped.device)
                    self.env.unwrapped.bpp.packed_obj_idx[i].append(torch.tensor(packed_obj_idx, device=self.env.unwrapped.device))
                    
            except Exception as e:
                print(f"Multiprocessing failed, falling back to sequential: {e}")
                self.mp_enabled = False
                # Fall back to sequential processing
                for i in range(actions.shape[0]):
                    # Analytically determine the z position of the object
                    curr_attitude = Attitude(0, 0, 0) if qz[i] == 0 and qw[i] == 1 else Attitude(0, 0, torch.pi / 2)
                    item_to_add = self.env.unwrapped.bpp.problems[i].items[int(actions[i, 1])]
                    item_to_add.rotate(curr_attitude)
                    item_to_add.calc_heightmap()
                    # Convert grid indices to actual coordinates
                    x_coord = int(torch.floor(x[i]).item())
                    y_coord = int(torch.floor(y[i]).item())
                    self.env.unwrapped.bpp.problems[i].container.add_item_topdown(item_to_add, x_coord, y_coord)
                    transform = Transform(
                        Position(self.env.unwrapped.bpp.problems[i].container.geometry.x_size - x_coord, self.env.unwrapped.bpp.problems[i].container.geometry.y_size - y_coord, item_to_add.position.z),
                        curr_attitude,
                    )
                    item_to_add.transform(transform)
                    self.env.unwrapped.bpp.problems[i].container.add_item(item_to_add)
                    z[i] = rotated_half_dim[i, 2] + item_to_add.position.z / 100
                    self.env.unwrapped.bpp.packed_obj_idx[i].append(actions[i, 1].type(torch.int64).cpu())
        else:
            # Sequential processing for single items or when multiprocessing is disabled
            for i in range(actions.shape[0]):
                # Analytically determine the z position of the object
                curr_attitude = Attitude(0, 0, 0) if qz[i] == 0 and qw[i] == 1 else Attitude(0, 0, torch.pi / 2)
                item_to_add = self.env.unwrapped.bpp.problems[i].items[int(actions[i, 1])]
                item_to_add.rotate(curr_attitude)
                item_to_add.calc_heightmap()
                # Convert grid indices to actual coordinates
                x_coord = int(torch.floor(x[i]).item())
                y_coord = int(torch.floor(y[i]).item())
                self.env.unwrapped.bpp.problems[i].container.add_item_topdown(item_to_add, x_coord, y_coord)
                transform = Transform(
                    Position(self.env.unwrapped.bpp.problems[i].container.geometry.x_size - x_coord, self.env.unwrapped.bpp.problems[i].container.geometry.y_size - y_coord, item_to_add.position.z),
                    curr_attitude,
                )
                item_to_add.transform(transform)
                self.env.unwrapped.bpp.problems[i].container.add_item(item_to_add)
                z[i] = rotated_half_dim[i, 2] + item_to_add.position.z / 100
                self.env.unwrapped.bpp.packed_obj_idx[i].append(actions[i, 1].type(torch.int64).cpu())
                if i == 0:
                    from packing3d import Display
                    display = Display(self.env.unwrapped.bpp.problems[i].box_size)
                    display.show3d(self.env.unwrapped.bpp.problems[i].container.geometry)
                    plt.savefig(f"container_{i}.png")
            
        print(f"Time taken to convert to pos_quat: {time.time() - start_time:.3f}s")
        pos_quat = torch.stack([actions[:, 0], actions[:, 1], x, y, z, qx, qy, qz, qw], dim=1)
        return pos_quat

    def step(self, actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        tote_ids = torch.arange(self.env.unwrapped.num_envs, device=self.env.unwrapped.device) % self.env.unwrapped.tote_manager.num_totes
        packable_objects = self.env.unwrapped.bpp.get_packable_object_indices(self.env.unwrapped.tote_manager.num_objects, self.env.unwrapped.tote_manager, torch.arange(self.env.unwrapped.num_envs, device=self.env.unwrapped.device), tote_ids)[0]
        actions = torch.cat(
            [
                tote_ids.unsqueeze(1).to(self.env.unwrapped.device),  # Destination tote IDs
                torch.tensor([row[0] for row in packable_objects], device=self.env.unwrapped.device).unsqueeze(1),  # Object indices
                actions,
            ], dim=1
        )
        actions = self._convert_to_pos_quat(actions)
        # clip actions
        if self.clip_actions is not None:
            actions = torch.clamp(actions, -self.clip_actions, self.clip_actions)
        # record step information
        obs_dict, rew, terminated, truncated, extras = self.env.unwrapped.step(actions)

        self.env.unwrapped.bpp.update_container_heightmap(self.env, torch.arange(self.env.unwrapped.num_envs, device=self.env.unwrapped.device), tote_ids)

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