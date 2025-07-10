# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import itertools

import isaaclab.utils.math as math_utils
import torch
from isaaclab.sim import schemas
from isaaclab.sim.schemas import schemas_cfg
from tote_consolidation.tasks.manager_based.pack.utils.tote_statistics import (
    ToteStatistics,
)


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
        self.dest_totes = None
        self.overfill_threshold = 0.2  # in meters
        self.max_objects_per_tote = 2

        tote_bbox = self.true_tote_dim * 0.01  # Convert cm to m for bounding box calculations
        for tote in self.tote_assets:
            tote_pose = tote.get_world_poses()[0][0] - env.scene.env_origins[0, :3]
            bounds = [
                (tote_pose[0] - tote_bbox[0] / 2, tote_pose[0] + tote_bbox[0] / 2),
                (tote_pose[1] - tote_bbox[1] / 2, tote_pose[1] + tote_bbox[1] / 2),
                (tote_pose[2], tote_pose[2] + tote_bbox[2]),
            ]
            self.tote_bounds.append(bounds)

        # Initialize statistics tracker
        self.stats = ToteStatistics(self.num_envs, self.num_totes, env.device)
        self.log_stats = True
        self.animate = True

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

        self.dest_totes = tote_ids
        # Remove objects from their original tote
        self.tote_to_obj[env_ids, :, object_ids] = 0
        # Mark objects as placed in the specified tote
        self.tote_to_obj[env_ids, tote_ids, object_ids] = 1

    def refill_source_totes(self, env, env_ids):
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
        while has_empty.any():
            first_empty = torch.argmax(empty_totes.float(), dim=1)

            # Create a mask for environments where first empty tote is a destination tote
            # Reshape dest_totes to ensure proper broadcasting
            if self.dest_totes is not None:
                dest_totes_reshaped = self.dest_totes.view(-1, 1) if self.dest_totes.dim() == 1 else self.dest_totes
                is_dest_tote = first_empty.unsqueeze(1) == dest_totes_reshaped
                is_dest_tote_any = is_dest_tote.any(dim=1) if is_dest_tote.dim() > 1 else is_dest_tote

                # Set first_empty to -1 if it's a destination tote
                first_empty = torch.where(is_dest_tote_any, torch.full_like(first_empty, -1), first_empty)

            # Update has_empty to exclude cases where first empty is a dest tote (now -1)
            has_empty = has_empty & (first_empty != -1)

            # Create empty tote tensor, -1 if no valid empty source tote
            empty_tote_tensor = torch.where(has_empty, first_empty, torch.full_like(first_empty, -1))

            # In all has-empty environments, teleport objects from reserve to the first empty tote
            if has_empty.any():
                if self.animate:
                    self._reappear_tote_animation(env, env_ids, has_empty, empty_tote_tensor)

                # Log source tote ejections
                if self.log_stats:
                    self.stats.log_source_tote_ejection(env_ids[has_empty])

                self.sample_and_place_objects_in_totes(
                    env, empty_tote_tensor[has_empty], env_ids[has_empty], self.max_objects_per_tote
                )
            # Check if all objects in each tote are zero
            empty_totes = torch.all(self.tote_to_obj[env_ids] == 0, dim=2)

            # Find first empty tote per environment (or -1 if none)
            has_empty = empty_totes.any(dim=1)

    def get_tote_fill_height(self, env, tote_ids, env_ids):
        """
        Calculate the fill height for specified totes in given environments.

        Args:
            env: Simulation environment object.
            tote_ids (torch.Tensor): Tensor containing tote IDs.
            env_ids (torch.Tensor): Tensor containing environment IDs.

        Returns:
            torch.Tensor: Fill height for the specified totes in the given environments.
        """
        # Get asset configurations for the specified environments
        assets = [env.scene[f"object{obj_id}"] for obj_id in range(self.num_objects)]
        max_z_positions = torch.zeros(len(env_ids), device=env.device)

        for asset in assets:
            obj_id = int(asset.cfg.prim_path.split("Object")[-1])

            # First check if object is in destination tote of any env to avoid unnecessary calculations
            object_in_dest_tote = self.tote_to_obj[env_ids, tote_ids, obj_id] == 1
            if not object_in_dest_tote.any():
                continue

            # Get positions for all environments
            asset_position = asset.data.root_state_w[:, :3] - env.scene.env_origins[:, :3]
            asset_orientation = asset.data.root_state_w[:, 3:7]

            # Get rotated bounding box
            obj_bbox = env.tote_manager.obj_bboxes[:, obj_id]
            rotated_dims = env.tote_manager._calculate_rotated_bounding_box(obj_bbox, asset_orientation, env.device)
            margin = rotated_dims / 2.0
            top_z_positions = asset_position[:, 2] + margin[:, 2]

            # Update max_z_positions only for relevant environments
            max_z_positions[object_in_dest_tote] = torch.maximum(
                max_z_positions[object_in_dest_tote], top_z_positions[object_in_dest_tote]
            )
        self.dest_totes = tote_ids
        self.refill_source_totes(env, env_ids)

        return max_z_positions

    def eject_destination_totes(self, env, tote_ids, env_ids):
        """
        Eject destination totes by resetting the objects in them to reserve.

        Args:
            env: Simulation environment object.
            tote_ids (torch.Tensor): Tensor containing tote IDs.
            env_ids (torch.Tensor): Tensor containing environment IDs.

        Returns:
            None
        """
        fill_heights = self.get_tote_fill_height(env, tote_ids, env_ids)

        # Check if any tote exceeds the overfill threshold
        overfilled_envs = fill_heights > self.overfill_threshold

        if not overfilled_envs.any():
            return

        # Animate tote ejection if enabled
        if self.animate:
            self._reappear_tote_animation(
                env,
                env_ids[overfilled_envs],
                torch.ones_like(env_ids[overfilled_envs], dtype=torch.bool),
                tote_ids[overfilled_envs],
            )

        # Log destination tote ejections
        if self.log_stats:
            self.stats.log_dest_tote_ejection(tote_ids[overfilled_envs], env_ids[overfilled_envs])

        assets_to_eject = []
        for env_id, tote_id in zip(env_ids[overfilled_envs], tote_ids[overfilled_envs]):
            # Get all objects in the overfilled tote
            objects_in_tote = torch.where(self.tote_to_obj[env_id, tote_id] == 1)[0]
            if objects_in_tote.numel() > 0:
                # Add to the list of assets to eject
                assets_to_eject.append([f"object{obj_id.item()}" for obj_id in objects_in_tote])

        for env_id, asset_names in zip(env_ids[overfilled_envs], assets_to_eject):
            # Reset the asset to reserve
            for asset_name in asset_names:
                asset = env.scene[asset_name]
                default_root_state = asset.data.default_root_state[env_id]
                default_root_state[:3] += env.scene.env_origins[env_id, :3]  # Adjust for environment origin
                asset.write_root_link_pose_to_sim(
                    default_root_state[:7], env_ids=torch.tensor([env_id], device=env.device)
                )
                asset.write_root_com_velocity_to_sim(
                    default_root_state[7:], env_ids=torch.tensor([env_id], device=env.device)
                )
                asset.set_visibility(False, env_ids=torch.tensor([env_id], device=env.device))

        if overfilled_envs.any():
            self.tote_to_obj[env_ids[overfilled_envs], tote_ids[overfilled_envs], :] = 0

    def update_object_positions_in_sim(self, env, objects, positions, orientations, cur_env):
        """
        Update object positions and orientations in the simulation.

        Args:
            env: Simulation environment object.
            objects (torch.Tensor or list): Object IDs or names to update.
            positions (torch.Tensor): Positions for each object [N, 3].
            orientations (torch.Tensor): Orientations for each object [N, 4].
            cur_env (int, torch.Tensor): Current environment ID(s).

        Returns:
            None
        """
        device = positions.device

        is_multi_env = hasattr(cur_env, "numel") and cur_env.numel() > 1

        # Update the object positions in the simulation
        for j, obj_id in enumerate(objects):
            # Get environment ID for this object
            if is_multi_env:
                env_id = cur_env[j].item()
            else:
                env_id = cur_env.item() if hasattr(cur_env, "item") else cur_env

            env_id_tensor = torch.tensor([env_id], device=device)

            # Get the asset based on object identifier type
            if isinstance(obj_id, str):
                asset = env.scene[obj_id]
            else:
                asset = env.scene[f"object{obj_id.item()}"]

            # Update prim path for the specific environment
            prim_path = asset.cfg.prim_path.replace("env_.*", f"env_{env_id}")

            # Modify physics properties and apply pose and velocity
            schemas.modify_rigid_body_properties(
                prim_path,
                schemas_cfg.RigidBodyPropertiesCfg(
                    kinematic_enabled=False,
                    disable_gravity=False,
                ),
            )

            # Apply position and orientation
            asset.write_root_link_pose_to_sim(
                torch.cat([positions[j], orientations[j]]),
                env_ids=env_id_tensor,
            )

            # Reset velocity
            asset.write_root_com_velocity_to_sim(
                torch.zeros(6, device=device),
                env_ids=env_id_tensor,
            )
            asset.set_visibility(True, env_ids=env_id_tensor)

    def sample_and_place_objects_in_totes(self, env, tote_ids, env_ids, max_objects_per_tote=None):
        """
        Sample objects from reserve and place them in the specified totes.

        Args:
            env: Simulation environment object.
            tote_ids (torch.Tensor): Tensor containing tote IDs to place objects in.
            env_ids (torch.Tensor): Tensor containing environment IDs.
            max_objects_per_tote (int, optional): Maximum number of objects to place in each tote.
                If None, a random number will be chosen based on available objects.

        Returns:
            bool: True if objects were placed, False otherwise.
        """
        # Get reserved objects
        reserve_objects = self.get_reserved_objs_idx(env_ids)
        reserve_objects_idx = torch.stack(torch.where(reserve_objects), dim=1)
        reserve_objects_grouped = [
            reserve_objects_idx[reserve_objects_idx[:, 0] == idx, 1].tolist() for idx in range(len(env_ids))
        ]

        max_available = reserve_objects.sum(dim=1).max().item()

        # Handle the case where there are no reserved objects
        if max_available <= 0:
            return False

        # Ensure tote_ids is a tensor with the right shape
        if not isinstance(tote_ids, torch.Tensor):
            tote_ids = torch.tensor(tote_ids, device=env_ids.device)

        if tote_ids.ndim == 0:
            tote_ids = tote_ids.unsqueeze(0)

        # Generate random number of objects to teleport or use specified max
        if max_objects_per_tote is None:
            num_objects_to_teleport = torch.randint(
                low=1,
                high=max_available + 1,  # +1 because randint is exclusive at the high end
                size=(len(env_ids),),
                device=env_ids.device,
            )
        else:
            num_objects_to_teleport = torch.full(
                (len(env_ids),), min(max_objects_per_tote, max_available), device=env_ids.device
            )

        # Sample the objects to teleport
        sampled_objects = []
        for i, num in enumerate(num_objects_to_teleport):
            if num > 0 and reserve_objects_grouped[i]:
                sampled_indices = torch.randperm(len(reserve_objects_grouped[i]), device=env_ids.device)[:num]
                sampled_objects.append(torch.tensor(reserve_objects_grouped[i], device=env_ids.device)[sampled_indices])
            else:
                sampled_objects.append(torch.tensor([], device=env_ids.device))

        for i, (cur_env, objects, tote_id) in enumerate(zip(env_ids, sampled_objects, tote_ids)):
            if objects.numel() > 0:
                # Sample positions above tote_bounds
                tote_bounds = self.tote_bounds[tote_id.item()]

                orientations_init = torch.tensor([1, 0, 0, 0], device=env_ids.device)
                orientations_to_apply = torch.tensor([1, 0, 0, 0], device=env_ids.device)

                # Extract tote boundaries
                x_min, x_max = tote_bounds[0]
                y_min, y_max = tote_bounds[1]
                z_min, z_max = tote_bounds[2]

                # Calculate rotated bounding boxes using the function
                orientations_to_apply_repeated = orientations_to_apply.repeat(objects.numel(), 1)
                rotated_dims = self._calculate_rotated_bounding_box(
                    self.obj_bboxes[cur_env, objects], orientations_to_apply_repeated, env_ids.device
                )

                repeated_orientations = torch.stack(
                    [math_utils.quat_mul(orientations_init, orientations_to_apply)] * objects.numel()
                )

                # Adjust placement bounds for each object based on its rotated dimensions
                margin = rotated_dims / 2.0
                adjusted_x_min = x_min + margin[:, 0]
                adjusted_x_max = x_max - margin[:, 0]
                adjusted_y_min = y_min + margin[:, 1]
                adjusted_y_max = y_max - margin[:, 1]
                adjusted_z_min = z_min
                adjusted_z_max = z_max

                # Generate random positions within adjusted bounds for all objects
                x_pos = (
                    torch.rand(objects.numel(), device=env_ids.device) * (adjusted_x_max - adjusted_x_min)
                    + adjusted_x_min
                )
                y_pos = (
                    torch.rand(objects.numel(), device=env_ids.device) * (adjusted_y_max - adjusted_y_min)
                    + adjusted_y_min
                )
                z_pos = (
                    torch.rand(objects.numel(), device=env_ids.device) * (adjusted_z_max - adjusted_z_min)
                    + adjusted_z_min
                    + self.true_tote_dim[2] * 0.01
                )

                # Stack positions into a single tensor
                positions = torch.stack([x_pos, y_pos, z_pos], dim=1)  # Shape: [num_objects, 3]
                positions += env.scene.env_origins[cur_env, :3]  # Adjust positions to the environment origin

                # Update object positions using the new function
                self.update_object_positions_in_sim(env, objects, positions, repeated_orientations, cur_env)

                # Place the sampled objects in the specified tote
                self.put_objects_in_tote(objects, tote_id, torch.tensor([cur_env], device=env_ids.device))
        return True

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

    def log_current_gcu(self, env_ids=None):
        """
        Log current GCU values for specified environments.

        Args:
            env_ids: Optional tensor of environment IDs to log for
        """
        if not self.log_stats:
            return

        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.obj_volumes.device)

        gcu_values = self.get_gcu(env_ids)
        self.stats.increment_step()
        self.stats.log_gcu(gcu_values, env_ids)

    def get_stats_summary(self):
        """Get a summary of all tracked statistics."""
        if self.log_stats:
            return self.stats.get_summary()
        return {}

    def save_stats(self, filepath):
        """Save statistics to a file."""
        if self.log_stats:
            self.stats.save_to_file(filepath)

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

    def _calculate_rotated_bounding_box(self, object_bboxes, orientations, device):
        """
        Calculate rotated bounding boxes for objects efficiently.

        Args:
            object_bboxes (torch.Tensor): Bounding boxes of objects [N, 3].
            orientations (torch.Tensor): Orientation quaternions [N, 4].
            device (torch.device): Device to use for tensors.

        Returns:
            torch.Tensor: Dimensions of rotated bounding boxes.
        """
        # Calculate rotated bounding boxes efficiently
        object_half_dims = object_bboxes[:, [1, 2, 0]] / 2.0 * 0.01  # cm to m

        # Create corner coordinates and compute rotated bounds in one batch operation
        corners = torch.tensor(list(itertools.product([-1, 1], repeat=3)), device=device).unsqueeze(
            0
        ) * object_half_dims.unsqueeze(1)
        rot_matrices = math_utils.matrix_from_quat(orientations)
        rotated_corners = torch.bmm(corners, rot_matrices.transpose(1, 2))

        # Get bounds of rotated boxes
        min_vals, _ = torch.min(rotated_corners, dim=1)
        max_vals, _ = torch.max(rotated_corners, dim=1)
        rotated_dims = max_vals - min_vals

        return rotated_dims

    def _step_in_sim(self, env, num_steps=1):
        """
        Step the simulation for a specified number of steps.

        Args:
            env: Simulation environment object.
            num_steps (int): Number of steps to perform in the simulation.
        """
        step = env._sim_step_counter
        for _ in range(num_steps):
            env.sim.step(render=False)
            if step % env.cfg.sim.render_interval == 0:
                env.sim.render()
            step += 1
            # update buffers at sim dt
            env.scene.update(dt=env.physics_dt)

    def _reappear_tote_animation(self, env, env_ids, eject_envs, eject_tote_ids):
        """
        Animate totes to disappear and reappear.

        Args:
            env: Simulation environment object.
            env_ids: Tensor containing environment IDs.
            eject_envs: Tensor of boolean flags indicating environments with empty totes.
            eject_tote_ids: Tensor containing IDs of empty totes.
        """
        self._step_in_sim(env, 20)
        # Animate tote to disappear and reappear
        for env_idx, tote_idx in zip(env_ids[eject_envs], eject_tote_ids[eject_envs]):
            if tote_idx != -1:
                # Get the asset for the empty tote
                tote_asset = env.scene[self.tote_keys[tote_idx.item()]]
                # Animate the tote to disappear and reappear
                tote_asset.set_visibilities([False], [env_idx.item()])

        self._step_in_sim(env, 20)

        for env_idx, tote_idx in zip(env_ids[eject_envs], eject_tote_ids[eject_envs]):
            if tote_idx != -1:
                tote_asset = env.scene[self.tote_keys[tote_idx.item()]]
                tote_asset.set_visibilities([True], [env_idx.item()])

    def get_packable_object_indices(self, num_objects, env_indices, tote_ids):
        """Get indices of objects that can be packed per environment.

        Args:
            num_objects: Number of objects per environment
            env_indices: Indices of environments to get packable objects for
            tote_ids: Destination tote IDs for each environment

        Returns:
            List of tensors containing packable object indices for each environment
        """
        num_envs = env_indices.shape[0]

        # Get objects that are reserved (already being picked up)
        reserved_objs = self.get_reserved_objs_idx(env_indices)

        # Get objects that are already in destination totes
        objs_in_dest = self.get_tote_objs_idx(tote_ids, env_indices)

        # Create a 2D tensor of object indices: shape (num_envs, num_objects)
        obj_indices = torch.arange(0, num_objects, device=env_indices.device).expand(num_envs, -1)

        # Compute mask of packable objects
        mask = (~reserved_objs & ~objs_in_dest).bool()

        # Use list comprehension to get valid indices per environment
        valid_indices = [obj_indices[i][mask[i]] for i in range(num_envs)]

        return valid_indices, mask
