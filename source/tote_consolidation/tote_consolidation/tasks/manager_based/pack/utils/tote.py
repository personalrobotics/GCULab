# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import torch


class Tote:
    def __init__(self, cfg, env):
        self.planning_tote_dim = torch.tensor([54, 40, 40], device=env.device)  # in cm
        self.true_tote_dim = torch.tensor([54, 35, 26], device=env.device)  # in cm
        self.tote_volume = torch.prod(self.true_tote_dim).item()
        self.obj_volumes = None
        self.obj_bboxes = None
        self.obj_T_bottomleft = None
        self.device = env.device
        self.num_envs = env.num_envs
        self.num_objects = cfg.num_object_per_env
        self.obj_in_tote = torch.zeros(self.num_envs, self.num_objects, dtype=torch.int32, device=self.device)

    def set_object_volume(self, obj_volumes):
        """Set object volumes."""
        self.obj_volumes = obj_volumes

    def set_object_bbox(self, obj_bboxes):
        """Set object bounding boxes."""
        self.obj_bboxes = obj_bboxes

    def set_object_T_bottomleft(self, obj_T_bottomleft):
        """Set object transformation matrices."""
        self.obj_T_bottomleft = obj_T_bottomleft

    def put_objects_in_totes(self, object_ids):
        """Mark specified objects as placed in the tote."""
        if self.obj_volumes is None:
            raise ValueError("Object volumes not set.")
        object_ids -= 1  # object IDs are 1-based
        rows = torch.arange(self.num_envs, device=self.device)
        self.obj_in_tote[rows, object_ids] = 1

    def get_gcus(self):
        """Compute GCU (tote utilization)."""
        if self.obj_volumes is None:
            raise ValueError("Object volumes not set.")
        obj_volumes = self.obj_volumes * self.obj_in_tote
        obj_volumes = torch.sum(obj_volumes, dim=1)
        obj_volumes = torch.clamp(obj_volumes, max=self.tote_volume)
        gcu = obj_volumes / self.tote_volume
        return gcu

    def reset(self):
        """Reset object tracking."""
        self.obj_in_tote.zero_()
        self.obj_volumes = None
        self.obj_bboxes = None
        self.obj_T_bottomleft = None
