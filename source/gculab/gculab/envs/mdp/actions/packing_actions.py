# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import isaaclab.sim as sim_utils
import isaaclab.utils.string as string_utils
import torch
from isaaclab.managers.action_manager import ActionTerm
from isaaclab.sim import schemas
from isaaclab.sim.schemas import schemas_cfg
from isaacsim.core.prims import XFormPrim
from geodude.tasks.manager_based.pack.utils.tote_helpers import (
    calculate_rotated_bounding_box,
)

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

    from . import actions_cfg


class PackingAction(ActionTerm):

    cfg: actions_cfg.PackingActionCfg
    """The configuration of the action term."""

    _asset: XFormPrim
    """The asset on which the placement action is applied."""

    _clip: torch.Tensor
    """The clip applied to the input action."""

    def __init__(self, cfg: actions_cfg.PackingActionCfg, env: ManagerBasedEnv) -> None:
        # initialize the action term
        super().__init__(cfg, env)

        # get pose of the asset
        self.place_obj_bottomLeft = (
            cfg.place_obj_bottomLeft
        )  # origin is bottom left of object placed at bottom left of tote
        self.true_tote_dim = self._env.tote_manager.true_tote_dim / 100
        tote_keys = sorted(
            [key for key in self._env.scene.keys() if key.startswith("tote")], key=lambda k: int(k.removeprefix("tote"))
        )
        self.num_totes = len(tote_keys)

        self._tote_assets = [self._env.scene[key] for key in tote_keys]

        # get poses of totes
        self._tote_assets_state = torch.stack([tote.get_world_poses()[0] for tote in self._tote_assets], dim=0)

        # create tensors for raw and processed actions
        self._raw_actions = torch.zeros(self.num_envs, 9, device=self.device)
        self._processed_actions = torch.zeros(self.num_envs, 9, device=self.device)

        # parse clip
        if self.cfg.clip is not None:
            if isinstance(cfg.clip, dict):
                self._clip = torch.tensor([[-float("inf"), float("inf")]], device=self.device).repeat(
                    self.num_envs, self.action_dim, 1
                )
                index_list, _, value_list = string_utils.resolve_matching_names_values(self.cfg.clip, self._joint_names)
                self._clip[:, index_list] = torch.tensor(value_list, device=self.device)
            else:
                raise ValueError(f"Unsupported clip type: {type(cfg.clip)}. Supported types are dict.")

    """
    Properties.
    """

    @property
    def action_dim(self) -> int:
        return 9

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions

    """
    Operations.
    """

    @profile
    def process_actions(self, actions: torch.Tensor):
        # store the raw actions
        self._raw_actions[:] = actions
        self._processed_actions = self._raw_actions.clone()

        # get the position and orientation
        tote_ids = self._processed_actions[:, 0].long()
        tote_states = self._tote_assets_state.permute(1, 0, 2)
        batch_indices = torch.arange(tote_ids.shape[0])

        tote_state = tote_states[batch_indices, tote_ids]

        # offset to center of the tote
        self._processed_actions[:, 2:5] += tote_state[:, :3].squeeze(1)

        if self.place_obj_bottomLeft:
            # offset to bottom left of the object
            self._processed_actions[:, 2:5] -= torch.tensor(
                [self.true_tote_dim[0] / 2, self.true_tote_dim[1] / 2, 0], device=self.device
            ).repeat(self.num_envs, 1)

            bbox_offset = torch.stack([
                self._env.tote_manager.get_object_bbox(env_idx, obj_idx.item())
                for env_idx, obj_idx in zip(batch_indices, actions[:, 1].long())
            ])

            rotated_half_dim = (
                calculate_rotated_bounding_box(
                    bbox_offset, self._processed_actions[:, 5:9].squeeze(1), device=self.device
                )
                / 2.0
            )
            # Get bounding box offset
            self._processed_actions[:, 2:5] += rotated_half_dim

        # compute the command
        # if self.cfg.clip is not None:
        #     self._processed_actions = torch.clamp(
        #         self._processed_actions, min=self._clip[:, :, 0], max=self._clip[:, :, 1]
        #     )

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        if env_ids is None:
            self._raw_actions[:] = torch.zeros_like(self._raw_actions)
        else:
            self._raw_actions[env_ids] = torch.zeros_like(self._raw_actions[env_ids])

    @profile
    def apply_actions(self):
        # first index is tote id
        tote_ids = self._processed_actions[:, 0].long()
        # second index is the object id
        object_ids = self._processed_actions[:, 1].long()
        # the rest are  position and orientation
        position = self._processed_actions[:, 2:5]
        # print("position", position)
        orientation = self._processed_actions[:, 5:9]

        self._env.tote_manager.last_action_pos_quat = self._processed_actions[:, 1:9]

        # Convert to list of objects
        objects = [f"object{obj_id.item()}" for obj_id in object_ids]

        # Update object positions using tote manager's method
        self._env.tote_manager.update_object_positions_in_sim(
            self._env, objects, position, orientation, cur_env=torch.arange(self.num_envs, device=self.device)
        )

        self._env.tote_manager.put_objects_in_tote(
            object_ids, tote_ids, env_ids=torch.arange(self.num_envs, device=self.device)
        )
