# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import torch


class ToteManager:
    """
    Manages tote-related operations in a multi-environment simulation.

    This class provides functionality to track objects in totes, compute tote utilization (GCUs),
    check for overfilling, and reset tote states. It assumes a simulation environment with multiple
    environments and totes.

    Attributes:
        tote_keys (list): Sorted list of tote keys in the scene.
        num_totes (int): Number of totes in the scene.
        tote_assets (list): List of tote assets in the scene.
        true_tote_dim (torch.Tensor): Dimensions of a tote in centimeters.
        tote_volume (float): Volume of a tote in cubic centimeters.
        num_envs (int): Number of environments in the simulation.
        num_objects (int): Number of objects per environment.
        obj_volumes (torch.Tensor): Volumes of objects in each environment.
        obj_bboxes (torch.Tensor): Bounding boxes of objects in each environment.
        tote_to_obj (torch.Tensor): Mapping of objects to totes in each environment.
        tote_bounds (list): Bounding box limits for each tote.
        overfill_threshold (float): Threshold for determining overfilling of a tote.
    """

    def __init__(self, cfg, env):
        """
        Initialize the ToteManager.

        Args:
            cfg: Configuration object containing simulation parameters.
            env: Simulation environment object.
        """
        self.tote_keys = sorted(
            [key for key in env.scene.keys() if key.startswith("tote")], key=lambda k: int(k.removeprefix("tote"))
        )
        self.num_totes = len(self.tote_keys)
        self.tote_assets = [env.scene[key] for key in self.tote_keys]

        self.true_tote_dim = torch.tensor([54, 35, 26], device=env.device)  # in cm
        self.tote_volume = torch.prod(self.true_tote_dim).item()
        self.num_envs = env.num_envs
        self.num_objects = cfg.num_object_per_env
        self.obj_volumes = torch.zeros(self.num_envs, self.num_objects, device=env.device)
        self.obj_bboxes = torch.zeros(self.num_envs, self.num_objects, 3, device=env.device)
        self.tote_to_obj = torch.zeros(
            self.num_envs, self.num_totes, self.num_objects, dtype=torch.int32, device=env.device
        )
        self.tote_bounds = []
        self.overfill_threshold = 0.3

        tote_bbox = self.true_tote_dim * 0.01  # Convert cm to m for bounding box calculations
        for tote in self.tote_assets:
            tote_pose = tote.get_world_poses()[0][0] - env.scene.env_origins[0, :3]
            tote_pose_tensor = torch.tensor(tote_pose, device=env.device)
            bounds = [
                (tote_pose_tensor[0] - tote_bbox[0] / 2, tote_pose_tensor[0] + tote_bbox[0] / 2),
                (tote_pose_tensor[1] - tote_bbox[1] / 2, tote_pose_tensor[1] + tote_bbox[1] / 2),
                (tote_pose_tensor[2], tote_pose_tensor[2] + tote_bbox[2]),
            ]
            self.tote_bounds.append(bounds)

    def set_object_volume(self, obj_volumes, env_ids):
        """
        Set object volumes for specific environments.

        Args:
            obj_volumes (torch.Tensor): Tensor containing object volumes.
            env_ids (torch.Tensor): Tensor containing environment IDs.
        """
        self.obj_volumes[env_ids] = obj_volumes

    def set_object_bbox(self, obj_bboxes, env_ids):
        """
        Set object bounding boxes for specific environments.

        Args:
            obj_bboxes (torch.Tensor): Tensor containing object bounding boxes.
            env_ids (torch.Tensor): Tensor containing environment IDs.
        """
        self.obj_bboxes[env_ids] = obj_bboxes

    def put_objects_in_tote(self, object_ids, tote_ids, env_ids):
        """
        Mark specified objects as placed in the specified tote for given environments.

        Args:
            object_ids (torch.Tensor): Tensor containing object IDs.
            tote_ids (torch.Tensor): Tensor containing tote IDs.
            env_ids (torch.Tensor): Tensor containing environment IDs.

        Raises:
            ValueError: If object volumes are not set.
        """
        if self.obj_volumes[env_ids].numel() == 0:
            raise ValueError("Object volumes not set.")
        # Create a mask for valid object IDs (non-zero)
        mask = object_ids > 0
        # Apply the mask to filter invalid object IDs while maintaining tensor dimensions
        filtered_object_ids = object_ids[mask]
        # Remove objects from their original tote
        self.tote_to_obj[env_ids, :, filtered_object_ids - 1] = 0  # object IDs are 1-based
        # Mark objects as placed in the specified tote
        valid_object_ids = filtered_object_ids[filtered_object_ids > 0]  # Filter out zero IDs
        self.tote_to_obj[env_ids, tote_ids, valid_object_ids - 1] = 1  # object IDs are 1-based

    def get_gcu(self, env_ids):
        """
        Compute GCUs (tote utilization) for all tote IDs for the specified environments.

        Args:
            env_ids (torch.Tensor): Tensor containing environment IDs.

        Returns:
            torch.Tensor: GCUs for each tote in the specified environments.

        Raises:
            ValueError: If object volumes are not set.
        """
        if self.obj_volumes[env_ids].numel() == 0:
            raise ValueError("Object volumes not set.")
        # Select the relevant totes for the given environments
        tote_selection = self.tote_to_obj[env_ids]  # Shape: [num_envs, num_totes, num_objects]
        # Multiply object volumes with tote selection and sum over the object dimension
        obj_volumes = torch.sum(
            self.obj_volumes[env_ids].unsqueeze(1) * tote_selection, dim=2
        )  # Shape: [num_envs, num_totes]
        # Compute GCUs as the ratio of used volume to total tote volume
        gcus = obj_volumes / self.tote_volume
        return gcus

    def get_fill_height(self, tote_ids, env_ids):
        """
        Compute fill height for the specified tote and environments.

        Args:
            tote_ids (torch.Tensor): Tensor containing tote IDs.
            env_ids (torch.Tensor): Tensor containing environment IDs.

        Raises:
            NotImplementedError: This method is not implemented yet.
        """
        raise NotImplementedError("Fill height calculation is not implemented yet.")

    def is_tote_overfilled(self, tote_ids, env_ids):
        """Check if the specified tote is overfilled for given environments."""
        raise NotImplementedError("Overfill check is not implemented yet.")

    def reset_tote(self, tote_ids, env_ids):
        """Reset object tracking for the specified tote and environments."""
        self.tote_to_obj[env_ids, tote_ids, :] = torch.zeros_like(
            self.tote_to_obj[env_ids, tote_ids, :]
        )  # Explicit reset

    def reset(self):
        """Reset object tracking for all totes across all environments."""
        self.tote_to_obj.zero_()
