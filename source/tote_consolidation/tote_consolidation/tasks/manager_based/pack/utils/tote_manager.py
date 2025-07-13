# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import itertools

import isaaclab.utils.math as math_utils
import torch
from isaaclab.sim import schemas
from isaaclab.sim.schemas import schemas_cfg
from tote_consolidation.tasks.manager_based.pack.utils.tote_helpers import (
    calculate_rotated_bounding_box,
    calculate_tote_bounds,
    generate_orientations,
    generate_positions,
    reappear_tote_animation,
    update_object_positions_in_sim,
)
from tote_consolidation.tasks.manager_based.pack.utils.tote_statistics import (
    ToteStatistics,
)


class ToteManager:
    """
    Manages totes and their contained objects across multiple simulation environments.

    Tracks object placement, calculates tote utilization (GCU), detects overfilling,
    and manages tote reset operations.

    Attributes:
        tote_keys: Sorted list of tote identifiers in the scene
        num_totes: Total number of totes
        tote_assets: List of tote asset references
        true_tote_dim: Physical dimensions of a tote (cm)
        tote_volume: Volume of a tote (cubic cm)
        num_envs: Number of simulation environments
        num_objects: Objects per environment
        obj_volumes: Object volumes by environment and object ID
        obj_bboxes: Object bounding boxes by environment and object ID
        tote_to_obj: Mapping between totes and objects (1 = object in tote)
        tote_bounds: Spatial boundaries of each tote
        overfill_threshold: Height threshold for tote overfilling
    """

    def __init__(self, cfg, env):
        """
        Initialize the ToteManager.

        Args:
            cfg: Configuration with simulation parameters
            env: Simulation environment
            animate_vis: Enable/disable animation effects
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
        self.obj_voxels = [[None for _ in range(self.num_objects)] for _ in range(self.num_envs)]
        self.tote_to_obj = torch.zeros(
            self.num_envs, self.num_totes, self.num_objects, dtype=torch.int32, device=env.device
        )
        self.tote_bounds = calculate_tote_bounds(self.tote_assets, self.true_tote_dim, env)
        self.dest_totes = torch.arange(self.num_envs, device=env.device) % self.num_totes  # Default to one tote per env
        self.overfill_threshold = 0.4  # in meters
        self.max_objects_per_tote = 2

        # Initialize statistics tracker
        self.stats = ToteStatistics(self.num_envs, self.num_totes, env.device)
        self.log_stats = True
        self.animate = cfg.animate_vis

    def set_object_volume(self, obj_volumes, env_ids):
        """
        Set object volumes for specified environments.

        Args:
            obj_volumes: Tensor of object volumes
            env_ids: Environment IDs to update
        """
        self.obj_volumes[env_ids] = obj_volumes

    def set_object_bbox(self, obj_bboxes, env_ids):
        """
        Set object bounding boxes for specified environments.

        Args:
            obj_bboxes: Tensor of object bounding boxes
            env_ids: Environment IDs to update
        """
        self.obj_bboxes[env_ids] = obj_bboxes

    def set_object_voxels(self, obj_voxels):
        """
        Set object voxel representations for specified environments.

        Args:
            obj_voxels: List of voxel representations for each environment
            env_ids: Environment IDs to update
        """
        self.obj_voxels = obj_voxels

    def put_objects_in_tote(self, object_ids, tote_ids, env_ids):
        """
        Mark objects as placed in specified totes.

        Args:
            object_ids: Objects to place
            tote_ids: Destination totes
            env_ids: Target environments

        Raises:
            ValueError: If object volumes aren't set
        """
        if self.obj_volumes[env_ids].numel() == 0:
            raise ValueError("Object volumes not set.")

        # Remove objects from their original tote
        self.tote_to_obj[env_ids, :, object_ids] = 0
        # Mark objects as placed in the specified tote
        self.tote_to_obj[env_ids, tote_ids, object_ids] = 1

    def _create_dest_totes_mask(self, empty_totes, env_ids):
        """
        Create a mask of destination totes for each environment.

        Args:
            empty_totes: Tensor indicating empty totes
            env_ids: Environment IDs

        Returns:
            Mask tensor for destination totes
        """
        dest_totes_mask = torch.zeros_like(empty_totes, dtype=torch.bool)
        for i, env_id in enumerate(env_ids):
            if self.dest_totes.dim() == 1:
                dest_totes_mask[i, self.dest_totes[env_id]] = True
            else:
                dest_totes_mask[i, self.dest_totes[env_id, :]] = True
        return dest_totes_mask

    def refill_source_totes(self, env, env_ids):
        """
        Refill empty source totes with objects from reserve.

        1. Identifies empty totes
        2. Refills empty source totes (excluding destination totes)
        3. Animates object appearance if enabled
        """
        # Check if all objects in each tote are zero
        empty_totes = torch.all(self.tote_to_obj[env_ids] == 0, dim=2)

        dest_totes_mask = self._create_dest_totes_mask(empty_totes, env_ids)

        # Mark destination totes as not empty to exclude them from refilling
        empty_totes = empty_totes & ~dest_totes_mask
        empty_totes_init = empty_totes.clone()

        # Find first empty tote per environment (or -1 if none)
        has_empty = empty_totes.any(dim=1)
        refilled = False
        outbound_gcus = self.get_gcu(env_ids)

        first_empty = torch.argmax(empty_totes.float(), dim=1)

        while has_empty.any():
            refilled = True

            # # Create empty tote tensor, -1 if no valid empty source tote
            # empty_tote_tensor = torch.where(has_empty, first_empty, torch.full_like(first_empty, -1))

            # In all has-empty environments, teleport objects from reserve to the first empty tote
            if self.animate:
                reappear_tote_animation(env, env_ids, has_empty, first_empty, self.tote_keys)

            self.sample_and_place_objects_in_totes(
                env, first_empty[has_empty], env_ids[has_empty], self.max_objects_per_tote
            )

            # Log source tote ejections
            if self.log_stats:
                self.stats.log_source_tote_ejection(env_ids[has_empty])

            # Check if all objects in each tote are zero
            empty_totes = torch.all(self.tote_to_obj[env_ids] == 0, dim=2)

            # Check if all objects in each tote are zero
            empty_totes = torch.all(self.tote_to_obj[env_ids] == 0, dim=2)

            # Create a mask of destination totes for each environment
            dest_totes_mask = self._create_dest_totes_mask(empty_totes, env_ids)
            # Mark destination totes as not empty to exclude them from refilling
            empty_totes = empty_totes & ~dest_totes_mask
            empty_totes_init = empty_totes.clone()

            # Find first empty tote per environment (or -1 if none)
            has_empty = empty_totes.any(dim=1)
            refilled = False
            outbound_gcus = self.get_gcu(env_ids)

            first_empty = torch.argmax(empty_totes.float(), dim=1)

        if refilled:
            inbound_gcus = self.get_gcu(env_ids)
            self.stats.log_tote_eject_gcus(inbound_gcus, outbound_gcus, totes_ejected=empty_totes_init)

    def get_tote_fill_height(self, env, tote_ids, env_ids):
        """
        Calculate the maximum height of objects in specified totes.

        Args:
            env: Simulation environment
            tote_ids: Totes to check
            env_ids: Target environments

        Returns:
            Maximum z-coordinate of objects in each tote
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
            rotated_dims = calculate_rotated_bounding_box(obj_bbox, asset_orientation, env.device)
            margin = rotated_dims / 2.0
            top_z_positions = asset_position[:, 2] + margin[:, 2]

            # Update max_z_positions only for relevant environments
            max_z_positions[object_in_dest_tote] = torch.maximum(
                max_z_positions[object_in_dest_tote], top_z_positions[object_in_dest_tote]
            )
        self.refill_source_totes(env, env_ids)

        return max_z_positions

    def eject_destination_totes(self, env, tote_ids, env_ids):
        """
        Reset overfilled destination totes by returning objects to reserve.

        Args:
            env: Simulation environment
            tote_ids: Destination totes
            env_ids: Target environments
        """
        fill_heights = self.get_tote_fill_height(env, tote_ids, env_ids)

        # Check if any tote exceeds the overfill threshold
        overfilled_envs = fill_heights > self.overfill_threshold

        if not overfilled_envs.any():
            return

        # Animate tote ejection if enabled
        if self.animate:
            reappear_tote_animation(
                env,
                env_ids[overfilled_envs],
                torch.ones_like(env_ids[overfilled_envs], dtype=torch.bool),
                tote_ids[overfilled_envs],
                self.tote_keys,
            )

        # Log destination tote ejections
        overfilled_totes = torch.zeros((self.num_envs, self.num_totes), dtype=torch.bool, device=env.device)
        overfilled_totes[overfilled_envs, tote_ids[overfilled_envs]] = True
        outbound_gcus = self.get_gcu(env_ids)
        if self.log_stats:
            self.stats.log_dest_tote_ejection(tote_ids[overfilled_envs], env_ids[overfilled_envs])
            self.stats.log_tote_eject_gcus(
                torch.zeros_like(outbound_gcus), outbound_gcus, totes_ejected=overfilled_totes
            )
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
                default_root_state[:3] += env.scene.env_origins[env_id, :3]
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
        Update object poses in the simulation.

        Args:
            env: Simulation environment
            objects: Object IDs/names to update
            positions: New positions [N, 3]
            orientations: New orientations [N, 4]
            cur_env: Environment ID(s)
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

    def sample_and_place_objects_in_totes(self, env, tote_ids, env_ids, max_objects_per_tote=None, min_separation=0.3):
        """
        Sample objects from reserve and place them in specified totes.

        Args:
            env: Simulation environment
            tote_ids: Target totes
            env_ids: Target environments
            max_objects_per_tote: Maximum objects to place per tote (random if None)
            min_separation: Minimum distance between objects in the same environment (in meters)

        Returns:
            True if objects were placed, False if no available objects
        """
        reserve_objects = self.get_reserved_objs_idx(env_ids)
        reserve_objects_idx = torch.stack(torch.where(reserve_objects), dim=1)
        reserve_objects_grouped = [
            reserve_objects_idx[reserve_objects_idx[:, 0] == idx, 1].tolist() for idx in range(len(env_ids))
        ]

        max_available = reserve_objects.sum(dim=1).max().item()
        if max_available <= 0:
            return False

        tote_ids = torch.tensor(tote_ids, device=env_ids.device) if not isinstance(tote_ids, torch.Tensor) else tote_ids
        tote_ids = tote_ids.unsqueeze(0) if tote_ids.ndim == 0 else tote_ids

        num_objects_to_teleport = (
            torch.randint(1, max_available + 1, (len(env_ids),), device=env_ids.device)
            if max_objects_per_tote is None
            else torch.full((len(env_ids),), min(max_objects_per_tote, max_available), device=env_ids.device)
        )

        sampled_objects = [
            (
                torch.tensor(reserve_objects_grouped[i], device=env_ids.device)[
                    torch.randperm(len(reserve_objects_grouped[i]), device=env_ids.device)[:num]
                ]
                if num > 0 and reserve_objects_grouped[i]
                else torch.tensor([], device=env_ids.device)
            )
            for i, num in enumerate(num_objects_to_teleport)
        ]

        for cur_env, objects, tote_id in zip(env_ids, sampled_objects, tote_ids):
            if objects.numel() > 0:
                tote_bounds = self.tote_bounds[tote_id.item()]
                # First generate orientations
                orientations = generate_orientations(objects)

                # Then generate positions based on orientations and rotated dimensions
                positions = generate_positions(
                    objects,
                    tote_bounds,
                    env.scene.env_origins[cur_env],
                    self.obj_bboxes[cur_env, objects],
                    orientations,
                    min_separation=min_separation,
                )

                update_object_positions_in_sim(env, objects, positions, orientations, cur_env)
                self.put_objects_in_tote(objects, tote_id, torch.tensor([cur_env], device=env_ids.device))

        return True

    def get_gcu(self, env_ids):
        """
        Calculate tote utilization (Gross Capacity Utilization).

        Args:
            env_ids: Target environments

        Returns:
            GCU values for each tote in specified environments

        Raises:
            ValueError: If object volumes aren't set
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
        Log current GCU values for tracking.

        Args:
            env_ids: Environments to log (all if None)
        """
        if not self.log_stats:
            return

        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.obj_volumes.device)

        gcu_values = self.get_gcu(env_ids)
        self.stats.log_gcu(gcu_values, env_ids)
        self.stats.increment_step()

    def get_stats_summary(self):
        """
        Get summary statistics for all tracked metrics.

        Returns:
            Dictionary of statistics
        """
        if self.log_stats:
            return self.stats.get_summary()
        return {}

    def save_stats(self, filepath):
        """
        Save statistics to file.

        Args:
            filepath: Path to save statistics
        """
        if self.log_stats:
            self.stats.save_to_file(filepath)

    def reset(self):
        """Reset object tracking for all totes."""
        self.tote_to_obj.zero_()

    def get_reserved_objs_idx(self, env_ids):
        """
        Get indices of objects in reserve (not in any tote).

        Args:
            env_ids: Target environments

        Returns:
            Boolean tensor indicating reserved objects
        """
        return self.tote_to_obj[env_ids].sum(dim=1) == 0

    def get_tote_objs_idx(self, tote_ids, env_ids):
        """
        Get indices of objects in specified totes.

        Args:
            tote_ids: Totes to check
            env_ids: Target environments

        Returns:
            Boolean tensor indicating objects in totes
        """
        return self.tote_to_obj[env_ids, tote_ids]
