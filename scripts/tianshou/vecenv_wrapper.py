# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym
from omegaconf import OmegaConf
import torch

from rsl_rl.env import VecEnv

from isaaclab.envs import DirectRLEnv, ManagerBasedRLEnv

import tianshou as ts
from tools import *

from envs.Packing.isaac_env import IsaacPackingEnv
from tote_consolidation.tasks.manager_based.pack.utils.tote_helpers import (
    calculate_rotated_bounding_box,
)
import matplotlib.pyplot as plt

class TianShouVecEnvWrapper(VecEnv):
    """Wraps around Isaac Lab environment for TianShou library

    To use asymmetric actor-critic, the environment instance must have the attributes :attr:`num_privileged_obs` (int).
    This is used by the learning agent to allocate buffers in the trajectory memory. Additionally, the returned
    observations should have the key "critic" which corresponds to the privileged observations. Since this is
    optional for some environments, the wrapper checks if these attributes exist. If they don't then the wrapper
    defaults to zero as number of privileged observations.

    .. caution::

        This class must be the last wrapper in the wrapper chain. This is because the wrapper does not follow
        the :class:`gym.Wrapper` interface. Any subsequent wrappers will need to be modified to work with this
        wrapper.

    Reference:
        https://github.com/leggedrobotics/rsl_rl/blob/master/rsl_rl/env/vec_env.py
    """

    def __init__(self, env: ManagerBasedRLEnv | DirectRLEnv, clip_actions: float | None = None, make_envs_args: OmegaConf | None = None):
        """Initializes the wrapper.

        Note:
            The wrapper calls :meth:`reset` at the start since the RSL-RL runner does not call reset.

        Args:
            env: The environment to wrap around.
            clip_actions: The clipping value for actions. If ``None``, then no clipping is done.

        Raises:
            ValueError: When the environment is not an instance of :class:`ManagerBasedRLEnv` or :class:`DirectRLEnv`.
        """
        # check that input is valid
        if not isinstance(env.unwrapped, ManagerBasedRLEnv) and not isinstance(env.unwrapped, DirectRLEnv):
            raise ValueError(
                "The environment must be inherited from ManagerBasedRLEnv or DirectRLEnv. Environment type:"
                f" {type(env)}"
            )
        # initialize the wrapper
        self.env = env
        self.clip_actions = clip_actions

        # store information required by wrapper
        self.num_envs = self.unwrapped.num_envs

        self.device = self.unwrapped.device
        self.max_episode_length = self.unwrapped.max_episode_length

        # obtain dimensions of the environment
        if hasattr(self.unwrapped, "action_manager"):
            self.num_actions = self.unwrapped.action_manager.total_action_dim
        else:
            self.num_actions = gym.spaces.flatdim(self.unwrapped.single_action_space)
        if hasattr(self.unwrapped, "observation_manager"):
            self.num_obs = self.unwrapped.observation_manager.group_obs_dim["policy"][0]
        else:
            self.num_obs = gym.spaces.flatdim(self.unwrapped.single_observation_space["policy"])
        # -- privileged observations
        if (
            hasattr(self.unwrapped, "observation_manager")
            and "critic" in self.unwrapped.observation_manager.group_obs_dim
        ):
            self.num_privileged_obs = self.unwrapped.observation_manager.group_obs_dim["critic"][0]
        elif hasattr(self.unwrapped, "num_states") and "critic" in self.unwrapped.single_observation_space:
            self.num_privileged_obs = gym.spaces.flatdim(self.unwrapped.single_observation_space["critic"])
        else:
            self.num_privileged_obs = 0

        # modify the action space to the clip range
        self._modify_action_space()

        # reset at the start since the RSL-RL runner does not call reset
        self.env.reset()

    def __str__(self):
        """Returns the wrapper name and the :attr:`env` representation string."""
        return f"<{type(self).__name__}{self.env}>"

    def __repr__(self):
        """Returns the string representation of the wrapper."""
        return str(self)

    """
    Properties -- Gym.Wrapper
    """

    @property
    def cfg(self) -> object:
        """Returns the configuration class instance of the environment."""
        return self.unwrapped.cfg

    @property
    def render_mode(self) -> str | None:
        """Returns the :attr:`Env` :attr:`render_mode`."""
        return self.env.render_mode

    @property
    def observation_space(self) -> gym.Space:
        """Returns the :attr:`Env` :attr:`observation_space`."""
        return self.env.observation_space

    @property
    def action_space(self) -> gym.Space:
        """Returns the :attr:`Env` :attr:`action_space`."""
        return self.env.action_space

    @classmethod
    def class_name(cls) -> str:
        """Returns the class name of the wrapper."""
        return cls.__name__

    @property
    def unwrapped(self) -> ManagerBasedRLEnv | DirectRLEnv:
        """Returns the base environment of the wrapper.

        This will be the bare :class:`gymnasium.Env` environment, underneath all layers of wrappers.
        """
        return self.env.unwrapped

    """
    Properties
    """

    def get_observations(self) -> tuple[torch.Tensor, dict]:
        """Returns the current observations of the environment."""
        if hasattr(self.unwrapped, "observation_manager"):
            obs_dict = self.unwrapped.observation_manager.compute()
        else:
            obs_dict = self.unwrapped._get_observations()
        return obs_dict["policy"], {"observations": obs_dict}

    @property
    def episode_length_buf(self) -> torch.Tensor:
        """The episode length buffer."""
        return self.unwrapped.episode_length_buf

    @episode_length_buf.setter
    def episode_length_buf(self, value: torch.Tensor):
        """Set the episode length buffer.

        Note:
            This is needed to perform random initialization of episode lengths in RSL-RL.
        """
        self.unwrapped.episode_length_buf = value

    """
    Operations - MDP
    """

    def seed(self, seed: int = -1) -> int:  # noqa: D102
        return self.unwrapped.seed(seed)

    def reset(self) -> tuple[torch.Tensor, dict]:  # noqa: D102
        # reset the environment
        obs_dict, _ = self.env.reset()

        # return observations
        return obs_dict["policy"], {"observations": obs_dict}

    def _convert_to_pos_quat(self, pos_rot: torch.Tensor, object_to_pack: list) -> torch.Tensor:
        orientation_one_hot = pos_rot[:, 2:]  # shape [batch, 2]
        orientation_idx = torch.argmax(orientation_one_hot, dim=1)  # shape [
        # Convert orientation index to quaternion
        # 0: 90° around z, 1: identity, 2: 90° around x, 3: 90° around y
        qx = torch.zeros_like(orientation_idx, dtype=torch.float32)
        qy = torch.zeros_like(orientation_idx, dtype=torch.float32)
        qz = torch.zeros_like(orientation_idx, dtype=torch.float32)
        qw = torch.ones_like(orientation_idx, dtype=torch.float32)

        # Set values for each orientation
        z_rot_mask = orientation_idx == 1

        # For 90° rotation around z (orientation 0)
        qx[z_rot_mask] = 0.7071068  # sin(π/4)
        qw[z_rot_mask] = 0.7071068  # cos(π/4)

        # Convert (x, y, theta) actions to (x, y, z, qx, qy, qz, qw)
        # Assume z=0 for placement (to be updated), and theta is in radians
        x = pos_rot[:, 0]
        y = pos_rot[:, 1]

        bbox_offset = self.env.unwrapped.tote_manager.obj_bboxes[
            torch.arange(pos_rot.shape[0], device=self.env.unwrapped.device), torch.tensor(object_to_pack, device=self.env.unwrapped.device)
        ][:, [0, 2, 1]]
        quats = torch.stack([qx, qy, qz, qw], dim=1)  # shape [batch, 4]
        rotated_dim = (
            calculate_rotated_bounding_box(
                bbox_offset, quats, device=self.env.unwrapped.device
            )
        )
        max_x = 51
        max_y = 34
        x = x / max_x
        y = y / max_y


        assert (x <= 1.0).all() and (y <= 1.0).all()
        x = (1 - x) * (self.env.unwrapped.tote_manager.true_tote_dim[0] / 100) - rotated_dim[:, 0] - 0.02
        y = y * (self.env.unwrapped.tote_manager.true_tote_dim[1] / 100) + 0.01

        # Compute z analytically for each sample in the batch using multiprocessing
        z = torch.zeros_like(x)
        return torch.stack([x, y, z, quats[:, 0], quats[:, 1], quats[:, 2], quats[:, 3]], dim=1), None, rotated_dim

    def _get_z_position_from_depth(self, image_obs: torch.Tensor, pos_rot: torch.Tensor, rotated_dim: torch.Tensor) -> torch.Tensor:
        """Get the z position from the depth image."""
        img_h = 34
        img_w = 51
        depth_img = image_obs.reshape(self.env.unwrapped.num_envs, img_h, img_w) / 100.0
        # plt.imshow(depth_img[0].cpu())
        # plt.savefig("depth_img.png")
        # plt.close()

        # Rescale x_pos and y_pos to the range of the depth image
        total_tote_x = self.env.unwrapped.tote_manager.true_tote_dim[0] / 100
        total_tote_y = self.env.unwrapped.tote_manager.true_tote_dim[1] / 100
        tote_x_m = self.env.unwrapped.tote_manager.true_tote_dim[0]
        tote_y_m = self.env.unwrapped.tote_manager.true_tote_dim[1]

        # Compute patch extents in pixel units by scaling world dimensions to pixel coordinates
        # The image covers the total tote dimensions, so scale object dimensions relative to total tote dimensions
        x_extent = torch.round((rotated_dim[:, 0] / total_tote_x) * tote_x_m).clamp(min=1).long()
        y_extent = torch.round((rotated_dim[:, 1] / total_tote_y) * tote_y_m).clamp(min=1).long()

        # Compute patch start/end indices, clamp to image bounds
        x0 = pos_rot[:, 0].clamp(0, tote_x_m)
        y0 = pos_rot[:, 1].clamp(0, tote_y_m)
        x1 = (x0 + x_extent).clamp(0, tote_x_m)
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
        plt.imshow(mask[0].cpu())
        plt.savefig("mask.png")
        plt.close()

        # Masked min: set out-of-patch values and zeros to inf, then take min
        depth_img_masked = depth_img.clone()
        depth_img_masked[~mask] = 0
        z_pos = depth_img_masked.view(depth_img.shape[0], -1).max(dim=1).values
        # plt.imshow(depth_img_masked[0].cpu())
        # plt.colorbar()
        # plt.savefig("depth_img_masked.png")
        # plt.close()

        # print("zpos: ", z_pos)
        z_pos = z_pos.clamp(min=0.0, max=0.4)
        return z_pos

    def step(self, pos_rot: torch.Tensor, image_obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        pos_rot = pos_rot[:, [1, 0, 2, 3]]
        # Swap x and y coordinates properly
        tote_ids = torch.zeros(self.env.unwrapped.num_envs, device=self.env.unwrapped.device).int()
        packable_objects = self.env.unwrapped.bpp.get_packable_object_indices(self.env.unwrapped.tote_manager.num_objects, self.env.unwrapped.tote_manager, torch.arange(self.env.unwrapped.num_envs, device=self.env.unwrapped.device), tote_ids)[0]
        object_to_pack = [row[0] for row in packable_objects]
        for i in range(self.env.unwrapped.num_envs):
            self.unwrapped.bpp.packed_obj_idx[i].append(torch.tensor([object_to_pack[i].item()]))
        
        actions, xy_pos_range, rotated_dim = self._convert_to_pos_quat(pos_rot, object_to_pack)
        # Get z_pos from depth image
        z_pos = self._get_z_position_from_depth(image_obs, pos_rot, rotated_dim)
        actions = torch.cat(
            [
                tote_ids.unsqueeze(1).to(self.env.unwrapped.device),  # Destination tote IDs
                torch.tensor(object_to_pack, device=self.env.unwrapped.device).unsqueeze(1),  # Object indices
                actions,
            ], dim=1
        )
        actions[:, 4] = z_pos

        # Convert actions to PackingActions
        # clip actions
        if self.clip_actions is not None:
            actions = torch.clamp(actions, -self.clip_actions, self.clip_actions)
        # record step information
        if not isinstance(actions, torch.Tensor):
            actions = torch.tensor(actions)
        obs_dict, rew, terminated, truncated, extras = self.env.step(actions)
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

    def close(self):  # noqa: D102
        return self.env.close()

    """
    Helper functions
    """

    def _modify_action_space(self):
        """Modifies the action space to the clip range."""
        if self.clip_actions is None:
            return

        # modify the action space to the clip range
        # note: this is only possible for the box action space. we need to change it in the future for other action spaces.
        self.env.unwrapped.single_action_space = gym.spaces.Box(
            low=-self.clip_actions, high=self.clip_actions, shape=(self.num_actions,)
        )
        self.env.unwrapped.action_space = gym.vector.utils.batch_space(
            self.env.unwrapped.single_action_space, self.num_envs
        )
