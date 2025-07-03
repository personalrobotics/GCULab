# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import torch
from isaaclab.sim import schemas
from isaaclab.sim.schemas import schemas_cfg

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

    def update_totes(self, env, dest_totes, env_ids):
        """
        1. Check if source totes are empty
        2. If source totes are empty, teleport a random number of objects from reserve
        3. If destination totes are full, make all objects in it reserve, then
            randomly sample reserve and place in the destination.
        """
        # Check if all objects in each tote are zero
        empty_totes = torch.all(self.tote_to_obj[env_ids] == 0, dim=2)

        # Find first empty tote per environment (or -1 if none)
        has_empty = empty_totes.any(dim=1)
        first_empty = torch.argmax(empty_totes.float(), dim=1)
        empty_tote_tensor = torch.where(has_empty, first_empty, torch.full_like(first_empty, -1))

        # in all has-empty environments, teleport a random number of objects from reserve
        if has_empty.any():
            # Get the indices of the empty totes
            empty_tote_indices = empty_tote_tensor[has_empty]
            # Get reserved objects
            reserve_objects = self.get_reserved_objs_idx(env_ids[has_empty])
            reserve_objects_idx = torch.stack(torch.where(reserve_objects), dim=1)
            reserve_objects_grouped = [
                reserve_objects_idx[reserve_objects_idx[:, 0] == idx, 1].tolist()
                for idx in range(len(env_ids[has_empty]))
            ]

            max_objects = reserve_objects.sum(dim=1).max().item()

            # Handle the case where there are no reserved objects
            if max_objects <= 0:
                return

            # Generate random number of objects to teleport (between 1 and max_objects)
            num_objects_to_teleport = torch.randint(
                low=1,
                high=max_objects + 1,  # +1 because randint is exclusive at the high end
                size=(empty_tote_indices.shape[0],),
                device=env_ids.device,
            )
            # Sample the objects to teleport
            sampled_objects = []
            for i, num in enumerate(num_objects_to_teleport):
                if num > 0 and reserve_objects_grouped[i]:
                    sampled_indices = torch.randperm(len(reserve_objects_grouped[i]), device=env_ids.device)[:num]
                    sampled_objects.append(
                        torch.tensor(reserve_objects_grouped[i], device=env_ids.device)[sampled_indices]
                    )
                else:
                    sampled_objects.append(torch.tensor([], device=env_ids.device))

            # Place the sampled objects in the empty totes
            for cur_env, objects, empty_tote_index in zip(env_ids[has_empty], sampled_objects, empty_tote_indices):
                if objects.numel() > 0:
                    # Sample positions above tote_bounds
                    tote_bounds = self.tote_bounds[empty_tote_index.item()]

                    # Calculate bounds for each dimension
                    x_min, x_max = tote_bounds[0]
                    y_min, y_max = tote_bounds[1]
                    z_min, z_max = tote_bounds[2]

                    # Generate random positions within bounds
                    x_pos = torch.rand(objects.numel(), device=env_ids.device) * (x_max - x_min) + x_min
                    y_pos = torch.rand(objects.numel(), device=env_ids.device) * (y_max - y_min) + y_min
                    z_pos = torch.rand(objects.numel(), device=env_ids.device) * (z_max - z_min) + z_min + self.true_tote_dim[2] * 0.01

                    # Stack positions into a single tensor
                    positions = torch.stack([x_pos, y_pos, z_pos], dim=1)  # Shape: [num_objects, 3]

                    positions += env.scene.env_origins[cur_env, :3]  # Adjust positions to the environment origin

                    # Update the object positions in the simulation
                    for i, obj_id in enumerate(objects):
                        asset = env.scene[f"object{obj_id.item() + 1}"]
                        prim_path = asset.cfg.prim_path.replace("env_.*", f"env_{cur_env.item()}")
                        schemas.modify_rigid_body_properties(
                            prim_path,
                            schemas_cfg.RigidBodyPropertiesCfg(
                                kinematic_enabled=False,
                                disable_gravity=False,
                            ),
                        )
                        asset.write_root_link_pose_to_sim(
                            torch.cat([positions[i], torch.tensor([0, 0, 0, 1], device=env_ids.device)]),
                            env_ids=torch.tensor([cur_env], device=env_ids.device),
                        )
                        asset.write_root_com_velocity_to_sim(
                            torch.zeros(6, device=env_ids.device),
                            env_ids=torch.tensor([cur_env], device=env_ids.device),
                        )
                    # for i in range(100):
                    #     env.scene.write_data_to_sim()
                    #     env.sim.step(render=False)
                    #     env.sim.render()
                    #     env.scene.update(dt=env.physics_dt)
                    print(f"[INFO]: Teleported {objects.numel()} objects to empty tote {empty_tote_index.item()} in environment {cur_env.item()}.")
                    # Place the sampled objects in the empty tote
                    self.put_objects_in_tote(
                        objects + 1, empty_tote_index, torch.tensor([cur_env], device=env_ids.device)
                    )

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

    def get_reserved_objs_idx(self, env_ids):
        """
        Get ids of objects that are reserved for the specified environments.

        Args:
            env_ids (torch.Tensor): Tensor containing environment IDs.

        Returns:
            torch.Tensor: Indices of objects reserved for the specified environments.
        """
        return self.tote_to_obj[env_ids].sum(dim=1) == 0

    def get_tote_objs_idx(self, tote_ids, env_ids):
        """
        Get ids of objects placed in the specified totes for the given environments.

        Args:
            tote_ids (torch.Tensor): Tensor containing tote IDs.
            env_ids (torch.Tensor): Tensor containing environment IDs.

        Returns:
            torch.Tensor: Indices of objects placed in the specified totes.
        """
        return self.tote_to_obj[env_ids, tote_ids]
