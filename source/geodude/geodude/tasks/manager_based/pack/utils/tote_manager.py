# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import itertools
import os
from datetime import datetime

import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
import torch
from isaaclab.sim import schemas
from isaaclab.sim.schemas import schemas_cfg
from pxr import Sdf, UsdGeom
from geodude.tasks.manager_based.pack.utils.tote_helpers import (  # Multiprocessing versions
    calculate_rotated_bounding_box,
    calculate_tote_bounds,
    generate_orientations_batched,
    generate_positions_batched_multiprocess_cuda,
    reappear_tote_animation,
    update_object_positions_in_sim_batched,
)
from geodude.tasks.manager_based.pack.utils.tote_statistics import (
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
        """
        self.tote_keys = sorted(
            [key for key in env.scene.keys() if key.startswith("tote")], key=lambda k: int(k.removeprefix("tote"))
        )
        self.num_totes = len(self.tote_keys)
        self.tote_assets = [env.scene[key] for key in self.tote_keys]
        self._tote_assets_state = torch.stack([tote.get_world_poses()[0] for tote in self.tote_assets], dim=0)

        self.true_tote_dim = torch.tensor([50, 34, 26], device=env.device)  # in cm (52 x 36 x 26)
        self.tote_volume = torch.prod(self.true_tote_dim).item()
        self.num_envs = env.num_envs
        self.num_objects = cfg.num_object_per_env  # in cm
        self.obj_asset_paths = [[None for _ in range(self.num_objects)] for _ in range(self.num_envs)]
        # Store unique object properties by asset path (volume, bbox, voxels)
        self.unique_object_properties = {}
        # Cache for fast volume lookups
        self._volume_cache = None
        # Cache for fast bbox lookups: (num_envs, num_objects, 3)
        self._bbox_cache = None
        self.tote_to_obj = torch.zeros(
            self.num_envs, self.num_totes, self.num_objects, dtype=torch.int32, device=env.device
        )
        self.tote_bounds = calculate_tote_bounds(self.tote_assets, self.true_tote_dim, env)
        # self.dest_totes = torch.arange(self.num_envs, device=env.device) % self.num_totes  # Default to one tote per env
        self.dest_totes = torch.zeros(self.num_envs, device=env.device).int()  # Default to one tote per env
        self.overfill_threshold = 0.3  # in meters
        self.max_objects_per_tote = 2

        self.last_pbrs = torch.zeros(self.num_envs, device=env.device, dtype=torch.float32)
        self.reset_pbrs = torch.zeros(self.num_envs, device=env.device, dtype=torch.bool)

        self.source_tote_ejected = torch.zeros(self.num_envs, dtype=torch.bool, device="cpu")
        self.log_stats = not cfg.disable_logging

        if self.log_stats:
            stats_dir = "stats"
            run_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            # create stats and run name directory if it does not exist
            if os.path.exists(stats_dir) is False:
                os.makedirs(stats_dir)
            run_name = f"Isaac-Pack-NoArm-v0_{run_dir}"
            # create run name directory
            run_path = os.path.join(stats_dir, run_name)
            if os.path.exists(run_path) is False:
                os.makedirs(run_path)
            save_path = os.path.join(run_path, "tote_stats.json")

            # Initialize statistics tracker
            self.stats = ToteStatistics(self.num_envs, self.num_totes, env.device, save_path, disable_logging=False)

        self.animate = cfg.animate_vis
        self.obj_settle_wait_steps = cfg.obj_settle_wait_steps
        self.last_action_pos_quat = torch.zeros(self.num_envs, 7, device=env.device)

        self.env = env
        self.device = env.device
        self._visibility_prims = None

    @property
    def collection(self):
        """The RigidObjectCollection managing all objects."""
        return self.env.scene["objects"]

    def _init_visibility_prims(self):
        """Cache prims for per-object visibility control (lazy init)."""
        if self._visibility_prims is not None:
            return
        collection = self.collection
        self._visibility_prims = []
        for prim_path_expr in collection._prim_paths:
            prims = sim_utils.find_matching_prims(prim_path_expr)
            self._visibility_prims.append(prims)

    @profile
    def set_object_visibility(self, visible, env_ids, object_ids):
        """Set visibility for specific objects in specific environments.

        Args:
            visible: Whether to make objects visible.
            env_ids: Environment indices (list or tensor).
            object_ids: Object indices (list or tensor).
        """
        self._init_visibility_prims()
        if isinstance(env_ids, torch.Tensor):
            env_ids = env_ids.detach().cpu().tolist()
        if isinstance(object_ids, torch.Tensor):
            object_ids = object_ids.detach().cpu().tolist()
        token = "inherited" if visible else "invisible"
        with Sdf.ChangeBlock():
            for obj_id in object_ids:
                prims = self._visibility_prims[obj_id]
                for env_id in env_ids:
                    UsdGeom.Imageable(prims[env_id]).GetVisibilityAttr().Set(token)

    def set_object_visibility_paired(self, visible, env_ids, object_ids):
        """Set visibility for paired (env_id, object_id) entries (not cross-product).

        Unlike set_object_visibility which iterates the cross-product of env_ids × object_ids,
        this method iterates zip(env_ids, object_ids) — each env_id[i] pairs with object_id[i].

        Args:
            visible: Whether to make objects visible.
            env_ids: Environment indices (list or tensor), paired with object_ids.
            object_ids: Object indices (list or tensor), paired with env_ids.
        """
        self._init_visibility_prims()
        if isinstance(env_ids, torch.Tensor):
            env_ids = env_ids.detach().cpu().tolist()
        if isinstance(object_ids, torch.Tensor):
            object_ids = object_ids.detach().cpu().tolist()
        token = "inherited" if visible else "invisible"
        with Sdf.ChangeBlock():
            for env_id, obj_id in zip(env_ids, object_ids):
                UsdGeom.Imageable(self._visibility_prims[obj_id][env_id]).GetVisibilityAttr().Set(token)

    def _write_poses_to_sim(self, env_ids, obj_ids, poses, velocities=None):
        """Batch-write poses and velocities to simulation via direct PhysX view.

        Args:
            env_ids: Environment indices tensor (N,).
            obj_ids: Object indices tensor (N,), paired with env_ids.
            poses: Pose tensor (N, 7) in wxyz quaternion convention.
            velocities: Optional velocity tensor (N, 6). Defaults to zeros.
        """
        collection = self.collection
        device = env_ids.device
        n = env_ids.shape[0]
        if n == 0:
            return

        env_ids_long = env_ids.long()
        obj_ids_long = obj_ids.long()

        # Compute flat PhysX view indices (object-major layout)
        flat_view_ids = obj_ids_long * collection.num_instances + env_ids_long

        # Convert quaternion wxyz -> xyzw for PhysX
        poses_xyzw = poses.clone()
        poses_xyzw[..., 3:] = math_utils.convert_quat(poses_xyzw[..., 3:], to="xyzw")

        if velocities is None:
            velocities = torch.zeros(n, 6, device=device)

        # PhysX requires full-view-sized tensors; indices select which rows to write
        total_bodies = collection.num_objects * collection.num_instances
        full_transforms = torch.zeros(total_bodies, 7, device=device)
        full_transforms[flat_view_ids] = poses_xyzw
        full_velocities = torch.zeros(total_bodies, 6, device=device)
        full_velocities[flat_view_ids] = velocities
        collection.root_physx_view.set_transforms(full_transforms, indices=flat_view_ids)
        collection.root_physx_view.set_velocities(full_velocities, indices=flat_view_ids)

        # Keep internal data buffers in sync
        if collection._data._object_link_pose_w.data is not None:
            collection._data._object_link_pose_w.data[env_ids_long, obj_ids_long] = poses
            collection._data._object_link_pose_w.timestamp = collection._data._sim_timestamp
        if collection._data._object_com_vel_w.data is not None:
            collection._data._object_com_vel_w.data[env_ids_long, obj_ids_long] = velocities
            collection._data._object_com_vel_w.timestamp = collection._data._sim_timestamp
        # Invalidate cached combined state tensors
        collection._data._object_state_w.timestamp = -1
        collection._data._object_link_state_w.timestamp = -1
        collection._data._object_com_state_w.timestamp = -1

    def set_object_asset_paths(self, obj_asset_paths, env_ids):
        """
        Set asset paths for objects in the simulation.
        """
        self.obj_asset_paths = obj_asset_paths

    def set_unique_object_properties(self, unique_properties):
        """
        Set unique object properties by asset path.

        Args:
            unique_properties: Dictionary mapping asset paths to (volume, bbox, voxels)
        """
        self.unique_object_properties = unique_properties
        # Build volume cache for fast GCU calculations
        self._build_volume_cache()
        self._build_bbox_cache()

    def get_object_volume(self, env_idx, obj_idx):
        """
        Get object volume by looking up asset path.

        Args:
            env_idx: Environment index
            obj_idx: Object index

        Returns:
            Object volume
        """
        asset_path = self.obj_asset_paths[env_idx][obj_idx]
        if asset_path is None or asset_path not in self.unique_object_properties:
            return 0.0
        return self.unique_object_properties[asset_path][0]

    def get_object_bbox(self, env_idx, obj_idx):
        """
        Get object bounding box by looking up asset path.

        Args:
            env_idx: Environment index
            obj_idx: Object index

        Returns:
            Object bounding box tensor
        """
        asset_path = self.obj_asset_paths[env_idx][obj_idx]
        if asset_path is None or asset_path not in self.unique_object_properties:
            return torch.zeros(3, device=self.device)
        bbox = self.unique_object_properties[asset_path][1]
        # Ensure bbox is on the correct device
        if isinstance(bbox, torch.Tensor):
            return bbox.to(self.device)
        return bbox

    def get_object_voxels(self, env_idx, obj_idx):
        """
        Get object voxels by looking up asset path.

        Args:
            env_idx: Environment index
            obj_idx: Object index

        Returns:
            Object voxels tensor or None
        """
        asset_path = self.obj_asset_paths[env_idx][obj_idx]
        if asset_path is None or asset_path not in self.unique_object_properties:
            return None
        voxels = self.unique_object_properties[asset_path][2]
        # Ensure voxels are on the correct device
        if isinstance(voxels, torch.Tensor):
            return voxels.to(self.device)
        return voxels

    def get_object_latents(self, env_idx, obj_idx):
        """
        Get object latents by looking up asset path.

        Args:
            env_idx: Environment index
            obj_idx: Object index

        Returns:
            Object latents tensor or None
        """
        asset_path = self.obj_asset_paths[env_idx][obj_idx]
        if asset_path is None or asset_path not in self.unique_object_properties:
            return None
        if isinstance(self.unique_object_properties[asset_path][3], torch.Tensor):
            return self.unique_object_properties[asset_path][3].to(self.device)
        return self.unique_object_properties[asset_path][3]

    def get_object_volumes_batch(self, env_ids, obj_indices):
        """
        Get object volumes for a batch of environments and objects.

        Args:
            env_ids: Environment indices
            obj_indices: Object indices

        Returns:
            Tensor of object volumes
        """
        volumes = torch.zeros(len(env_ids), len(obj_indices), device=self.device)
        for i, env_idx in enumerate(env_ids):
            for j, obj_idx in enumerate(obj_indices):
                volumes[i, j] = self.get_object_volume(env_idx, obj_idx)
        return volumes

    def get_object_bboxes_batch(self, env_ids, obj_indices):
        """
        Get object bounding boxes for a batch of environments and objects.

        Args:
            env_ids: Environment indices
            obj_indices: Object indices

        Returns:
            Tensor of object bounding boxes
        """
        bboxes = torch.zeros(len(env_ids), len(obj_indices), 3, device=self.device)
        for i, env_idx in enumerate(env_ids):
            for j, obj_idx in enumerate(obj_indices):
                bboxes[i, j] = self.get_object_bbox(env_idx, obj_idx)
        return bboxes

    def get_object_latents_batch(self, env_ids, obj_indices):
        """
        Get object latents for a batch of environments and objects.

        Args:
            env_ids: Environment indices
            obj_indices: Object indices

        Returns:
            Tensor of object latents
        """
        latents = torch.zeros(len(env_ids), len(obj_indices), 512, 8, device=self.device)
        for i, env_idx in enumerate(env_ids):
            for j, obj_idx in enumerate(obj_indices):
                latents[i, j] = self.get_object_latents(env_idx, obj_idx)
        return latents

    def _build_volume_cache(self):
        """Build a cache for fast volume lookups."""
        if self._volume_cache is not None:
            return

        self._volume_cache = torch.zeros(self.num_envs, self.num_objects, device=self.device, dtype=torch.float32)
        for env_idx in range(self.num_envs):
            for obj_idx in range(self.num_objects):
                asset_path = self.obj_asset_paths[env_idx][obj_idx]
                if asset_path is not None and asset_path in self.unique_object_properties:
                    volume = self.unique_object_properties[asset_path][0]
                    self._volume_cache[env_idx, obj_idx] = volume
                    print("asset_path ", asset_path)
                    print("bbox ", self.unique_object_properties[asset_path][1])
                    print("volume ", volume)

    def _build_bbox_cache(self):
        """Build a cache for fast bbox lookups: (num_envs, num_objects, 3)."""
        if self._bbox_cache is not None:
            return

        self._bbox_cache = torch.zeros(self.num_envs, self.num_objects, 3, device=self.device, dtype=torch.float32)
        for env_idx in range(self.num_envs):
            for obj_idx in range(self.num_objects):
                asset_path = self.obj_asset_paths[env_idx][obj_idx]
                if asset_path is not None and asset_path in self.unique_object_properties:
                    bbox = self.unique_object_properties[asset_path][1]
                    if isinstance(bbox, torch.Tensor):
                        self._bbox_cache[env_idx, obj_idx] = bbox.to(self.device)
                    else:
                        self._bbox_cache[env_idx, obj_idx] = torch.tensor(bbox, device=self.device, dtype=torch.float32)

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
        if not self.unique_object_properties:
            raise ValueError("Object properties not set.")

        # Remove objects from their original tote
        self.tote_to_obj[env_ids, :, object_ids] = 0
        # Mark objects as placed in the specified tote
        self.tote_to_obj[env_ids, tote_ids, object_ids] = 1

    def put_objects_in_tote_batched(self, all_object_ids, all_tote_ids, env_ids):
        """
        Mark objects as placed in specified totes for multiple environments in batch.

        Args:
            all_object_ids: List of object tensors for each environment
            all_tote_ids: List of tote tensors for each environment
            env_ids: Target environments

        Raises:
            ValueError: If object volumes aren't set
        """
        if not self.unique_object_properties:
            raise ValueError("Object properties not set.")

        # Vectorized processing: flatten all valid (env, object, tote) indices
        # Build lists of valid env indices, object ids, and tote ids
        env_indices = []
        object_indices = []
        tote_indices = []

        for env_idx, (object_ids, tote_ids) in enumerate(zip(all_object_ids, all_tote_ids)):
            if object_ids.numel() > 0:
                n = object_ids.numel()
                env_indices.append(env_ids[env_idx].expand(n))
                object_indices.append(object_ids)
                tote_indices.append(tote_ids)

        if len(env_indices) == 0:
            return

        env_indices = torch.cat(env_indices)
        object_indices = torch.cat(object_indices)
        tote_indices = torch.cat(tote_indices)

        # Remove objects from their original tote (set all tote rows for these objects to 0)
        self.tote_to_obj[env_indices, :, object_indices] = 0

        # Mark objects as placed in the specified tote
        self.tote_to_obj[env_indices, tote_indices, object_indices] = 1

    def _create_dest_totes_mask(self, empty_totes, env_ids):
        """
        Create a mask of destination totes for each environment (vectorized).

        Args:
            empty_totes: Tensor of shape (num_envs, num_totes)
            env_ids: Tensor of environment IDs (num_envs,)

        Returns:
            Mask tensor for destination totes (same shape as empty_totes)
        """
        num_envs, num_totes = empty_totes.shape
        dest_totes_mask = torch.zeros_like(empty_totes, dtype=torch.bool)

        if self.dest_totes.dim() == 1:
            # self.dest_totes: shape (total_totes,), one dest tote per env
            # Gather dest tote indices for the given env_ids
            dest_indices = self.dest_totes[env_ids]  # (num_envs,)
            dest_totes_mask[torch.arange(num_envs), dest_indices] = True
        else:
            # self.dest_totes: shape (num_envs_total, max_dest_per_env)
            dest_indices = self.dest_totes[env_ids]  # (num_envs, max_dest_per_env)
            env_range = torch.arange(num_envs).unsqueeze(1)  # (num_envs, 1)
            dest_totes_mask[env_range, dest_indices] = True  # broadcasting

        return dest_totes_mask

    @profile
    def refill_source_totes(self, env_ids):
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

        # Find first empty tote per environment (or -1 if none)
        has_empty = empty_totes.any(dim=1)
        refilled = False
        outbound_gcus = self.get_gcu(env_ids)

        first_empty = torch.argmax(empty_totes.float(), dim=1)
        self.source_tote_ejected[env_ids[has_empty]] = True

        loop_iter = 0
        while has_empty.any():
            refilled = True
            # In all has-empty environments, teleport objects from reserve to the first empty tote
            if self.animate:
                reappear_tote_animation(self.env, env_ids, has_empty, first_empty, self.tote_keys)

            self.sample_and_place_objects_in_totes(
                self.env, first_empty[has_empty], env_ids[has_empty], self.max_objects_per_tote
            )

            # Log source tote ejections
            if self.log_stats:
                self.stats.log_source_tote_ejection(env_ids[has_empty])

            # Check if all objects in each tote are zero
            empty_totes = torch.all(self.tote_to_obj[env_ids] == 0, dim=2)

            # Create a mask of destination totes for each environment
            dest_totes_mask = self._create_dest_totes_mask(empty_totes, env_ids)
            # Mark destination totes as not empty to exclude them from refilling
            empty_totes = empty_totes & ~dest_totes_mask

            # Find first empty tote per environment (or -1 if none)
            has_empty = empty_totes.any(dim=1)
            refilled = False
            outbound_gcus = self.get_gcu(env_ids)
            first_empty = torch.argmax(empty_totes.float(), dim=1)
            loop_iter += 1

        if refilled:
            inbound_gcus = self.get_gcu(env_ids)
            self.stats.log_tote_eject_gcus(inbound_gcus, outbound_gcus, totes_ejected=empty_totes)

    def get_tote_fill_height(self, tote_ids, env_ids, heightmaps=None):
        """
        Calculate the maximum height of objects in specified totes.

        Args:
            env: Simulation environment
            tote_ids: Totes to check
            env_ids: Target environments
            heightmaps: Heightmaps of the scene

        Returns:
            Maximum z-coordinate of objects in each tote
        """
        if heightmaps is not None:
            # heightmaps[env_ids] shape: (num_envs, H, W, 1)
            min_height = 20 - torch.amin(heightmaps[env_ids], dim=(1, 2, 3))  # shape: (num_envs,)
            return min_height
        else:
            collection = self.collection
            # Bulk read all object poses: (num_envs, num_objects, 7)
            all_poses = collection.data.object_link_pose_w
            all_positions = all_poses[:, :, :3] - self.env.scene.env_origins[:, :3].unsqueeze(1)
            all_orientations = all_poses[:, :, 3:7]

            max_z_positions = torch.zeros(self.env.num_envs, device=self.env.device)

            for obj_id in range(self.num_objects):
                # First check if object is in destination tote of any self.env to avoid unnecessary calculations
                object_in_dest_tote = self.tote_to_obj[torch.arange(self.env.num_envs), tote_ids, obj_id] == 1
                if not object_in_dest_tote.any():
                    continue

                # Get positions/orientations for this object across all envs
                asset_position = all_positions[:, obj_id]  # (num_envs, 3)
                asset_orientation = all_orientations[:, obj_id]  # (num_envs, 4)

                # Get rotated bounding box
                obj_bbox = torch.stack(
                    [self.get_object_bbox(env_idx, obj_id) for env_idx in range(self.env.num_envs)]
                )
                rotated_dims = calculate_rotated_bounding_box(obj_bbox, asset_orientation, self.env.device)
                margin = rotated_dims / 2.0
                top_z_positions = asset_position[:, 2] + margin[:, 2]

                # Update max_z_positions only for relevant environments
                max_z_positions[object_in_dest_tote] = torch.maximum(
                    max_z_positions[object_in_dest_tote], top_z_positions[object_in_dest_tote]
                )

            return max_z_positions[env_ids]

    @profile
    def eject_totes(self, tote_ids, env_ids, is_dest=True, overfill_check=True, heightmaps=None):
        """
        Reset overfilled destination totes by returning objects to reserve.

        Args:
            env: Simulation environment
            tote_ids: Destination totes
            env_ids: Target environments
            is_dest: Flag indicating if the totes are destination totes
            overfill_check: Flag to enable/disable overfill checking
        """
        fill_heights = self.get_tote_fill_height(tote_ids, env_ids, heightmaps)

        # Check if any tote exceeds the overfill threshold
        if overfill_check:
            overfilled_envs = fill_heights > self.overfill_threshold
        else:
            # Mock that the desired totes are overfilled, so that they are ejected
            overfilled_envs = torch.ones_like(fill_heights, dtype=torch.bool)

        if not overfilled_envs.any():
            return overfilled_envs

        # Animate tote ejection if enabled
        if self.animate:
            reappear_tote_animation(
                self.env,
                env_ids[overfilled_envs],
                torch.ones_like(env_ids[overfilled_envs], dtype=torch.bool),
                tote_ids[overfilled_envs],
                self.tote_keys,
            )

        # # Log destination tote ejections
        overfilled_totes = torch.zeros((self.num_envs, self.num_totes), dtype=torch.bool, device=self.env.device)
        overfilled_totes[env_ids[overfilled_envs], tote_ids[overfilled_envs]] = True
        overfilled_totes = overfilled_totes[env_ids]
        outbound_gcus = self.get_gcu(env_ids)
        if self.log_stats:
            if is_dest:
                # for env_idx, problem in zip(env_ids[overfilled_envs].tolist(), [self.env.unwrapped.bpp.problems[i.item()] for i in env_ids[overfilled_envs]]):
                #     print("logging dest tote for env_idx:", env_idx)
                #     self.env.unwrapped.bpp.update_container_heightmap(
                #         self.env, torch.tensor([env_idx], device=self.env.device), torch.zeros((self.num_envs), device=self.env.device).int()
                #     )
                #     self.stats.log_container(env_idx, problem.container)
                self.stats.log_dest_tote_ejection(tote_ids[overfilled_envs], env_ids[overfilled_envs])
            self.stats.log_tote_eject_gcus(
                torch.zeros_like(outbound_gcus), outbound_gcus, totes_ejected=overfilled_totes
            )

        # Collect all (env_id, obj_id) pairs to eject — vectorized batch lookup
        overfilled_env_ids = env_ids[overfilled_envs]
        overfilled_tote_ids = tote_ids[overfilled_envs]

        # Batch lookup: get object mask for all overfilled (env, tote) pairs at once
        obj_masks = self.tote_to_obj[overfilled_env_ids, overfilled_tote_ids]  # (N, num_objects)

        # Find all (pair_idx, obj_id) where mask == 1
        pair_idx, eject_obj_tensor = torch.where(obj_masks == 1)

        # Map pair_idx back to env_ids
        eject_env_tensor = overfilled_env_ids[pair_idx]

        if eject_env_tensor.numel() > 0:
            collection = self.collection

            # Get default states for all objects being ejected
            default_states = collection.data.default_object_state[eject_env_tensor.long(), eject_obj_tensor.long()].clone()
            default_states[:, :3] += self.env.scene.env_origins[eject_env_tensor.long(), :3]

            # Batch write poses and velocities
            self._write_poses_to_sim(eject_env_tensor, eject_obj_tensor, default_states[:, :7], default_states[:, 7:])

            # Set visibility off
            self.set_object_visibility_paired(False, eject_env_tensor, eject_obj_tensor)

        if overfilled_envs.any():
            self.tote_to_obj[env_ids[overfilled_envs], tote_ids[overfilled_envs], :] = 0
        return overfilled_envs

    @profile
    def update_object_positions_in_sim(self, env, objects, positions, orientations, cur_env):
        """
        Update object poses in the simulation using batched PhysX writes.

        Args:
            env: Simulation environment
            objects: Object IDs (tensor of ints, list of ints, or list of strings like "object5")
            positions: New positions [N, 3]
            orientations: New orientations [N, 4]
            cur_env: Environment ID(s)
        """
        device = positions.device

        # Fast path: both objects and cur_env are already tensors — skip the loop
        if isinstance(objects, torch.Tensor) and isinstance(cur_env, torch.Tensor) and cur_env.numel() > 1:
            n = objects.shape[0]
            env_ids_tensor = cur_env.long().to(device)
            obj_ids_tensor = objects.long().to(device)
        else:
            # Slow fallback for mixed types (lists, strings, single env)
            is_multi_env = hasattr(cur_env, "numel") and cur_env.numel() > 1
            n = len(objects) if not isinstance(objects, torch.Tensor) else objects.shape[0]
            env_ids_list = []
            obj_ids_list = []

            for j in range(n):
                obj_id = objects[j]
                if is_multi_env:
                    env_id = cur_env[j].item()
                else:
                    env_id = cur_env.item() if hasattr(cur_env, "item") else cur_env

                if isinstance(obj_id, str):
                    obj_idx = int(obj_id.replace("object", ""))
                elif isinstance(obj_id, torch.Tensor):
                    obj_idx = obj_id.item()
                else:
                    obj_idx = int(obj_id)

                env_ids_list.append(env_id)
                obj_ids_list.append(obj_idx)

            env_ids_tensor = torch.tensor(env_ids_list, device=device, dtype=torch.long)
            obj_ids_tensor = torch.tensor(obj_ids_list, device=device, dtype=torch.long)

        # Build poses tensor (N, 7)
        poses = torch.cat([positions[:n], orientations[:n]], dim=-1)

        # Batch write to simulation
        self._write_poses_to_sim(env_ids_tensor, obj_ids_tensor, poses)

        # Set visibility
        self.set_object_visibility_paired(True, env_ids_tensor, obj_ids_tensor)

    @profile
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
        reserve_objects = self.get_reserved_objs_idx(env_ids)  # (num_envs, num_objects) bool

        max_available = reserve_objects.sum(dim=1).max().item()
        if max_available <= 0:
            raise ValueError(
                "No available objects to sample from reserve. Increase num_object_per_env to increase the number of"
                " objects in the reserve."
            )

        tote_ids = torch.tensor(tote_ids, device=env_ids.device) if not isinstance(tote_ids, torch.Tensor) else tote_ids
        tote_ids = tote_ids.unsqueeze(0) if tote_ids.ndim == 0 else tote_ids

        K = min(max_objects_per_tote, max_available) if max_objects_per_tote is not None else max_available

        # --- Vectorized random sampling (replaces per-env loops) ---
        # Assign random keys to reserve objects, -1 to non-reserve
        rand_keys = torch.rand(len(env_ids), self.num_objects, device=self.device)
        rand_keys[~reserve_objects] = -1.0
        # topk picks K highest-random-key objects per env
        _, sampled_indices = rand_keys.topk(K, dim=1)  # (num_envs, K)

        # Per-env number of objects to teleport
        available_per_env = reserve_objects.sum(dim=1)  # (num_envs,)
        if max_objects_per_tote is None:
            # Original behavior: random count per env
            num_objects_to_teleport = torch.randint(1, max_available + 1, (len(env_ids),), device=self.device)
            num_objects_to_teleport = torch.min(num_objects_to_teleport, available_per_env)
        else:
            num_objects_to_teleport = available_per_env.clamp(max=K)

        valid_mask = torch.arange(K, device=self.device).unsqueeze(0) < num_objects_to_teleport.unsqueeze(1)

        if self.animate:
            # --- Animate path: convert to per-env lists for existing helpers ---
            sampled_objects = [
                sampled_indices[i, :valid_mask[i].sum()] for i in range(len(env_ids))
            ]

            # Collect tote bounds
            all_tote_bounds = [self.tote_bounds[tote_id.item()] for tote_id in tote_ids]

            # Collect object bboxes
            all_obj_bboxes = []
            for cur_env, objects in zip(env_ids, sampled_objects):
                if objects.numel() > 0:
                    bboxes = torch.stack([self.get_object_bbox(cur_env, obj.item()) for obj in objects])
                    all_obj_bboxes.append(bboxes)
                else:
                    all_obj_bboxes.append(torch.empty(0, 3, device=self.device))

            # Collect environment origins
            all_env_origins = [env.scene.env_origins[cur_env] for cur_env in env_ids]

            # Generate orientations
            all_orientations = generate_orientations_batched(sampled_objects, device=env_ids.device)

            # Generate positions
            all_positions = generate_positions_batched_multiprocess_cuda(
                sampled_objects,
                all_tote_bounds,
                all_env_origins,
                all_obj_bboxes,
                all_orientations,
                min_separation=min_separation,
            )

            # Update object positions in simulation
            update_object_positions_in_sim_batched(env, sampled_objects, all_positions, all_orientations, env_ids)

            # Update tote tracking via per-env lists
            all_tote_ids = [
                torch.full((obj.shape[0],), tid.item(), device=self.device) for obj, tid in zip(sampled_objects, tote_ids)
            ]
            self.put_objects_in_tote_batched(sampled_objects, all_tote_ids, env_ids)
        else:
            # --- Fast path: direct tensor update (no animation) ---
            env_broadcast = env_ids.unsqueeze(1).expand_as(sampled_indices)
            tote_broadcast = tote_ids.unsqueeze(1).expand_as(sampled_indices)

            flat_env = env_broadcast[valid_mask]
            flat_obj = sampled_indices[valid_mask]
            flat_tote = tote_broadcast[valid_mask]

            self.tote_to_obj[flat_env, :, flat_obj] = 0
            self.tote_to_obj[flat_env, flat_tote, flat_obj] = 1

        return True

    def get_gcu(self, env_ids):
        """
        Calculate tote utilization (Gross Capacity Utilization).

        Args:
            env_ids: Target environments

        Returns:
            GCU values for each tote in specified environments

        Raises:
            ValueError: If object properties aren't set
        """
        if not self.unique_object_properties:
            raise ValueError("Object properties not set.")

        # Build volume cache if not already built
        if self._volume_cache is None:
            self._build_volume_cache()

        # Select the relevant totes for the given environments
        tote_selection = self.tote_to_obj[env_ids]  # Shape: [num_envs, num_totes, num_objects]

        # Get volumes for the specified environments using pre-computed cache
        obj_volumes = self._volume_cache[env_ids]  # Shape: [num_envs, num_objects]

        # Expand volumes to match tote selection shape and multiply
        obj_volumes_expanded = obj_volumes.unsqueeze(1).expand(
            -1, self.num_totes, -1
        )  # Shape: [num_envs, num_totes, num_objects]

        # Multiply object volumes with tote selection and sum over the object dimension
        total_volumes = torch.sum(obj_volumes_expanded * tote_selection, dim=2)  # Shape: [num_envs, num_totes]

        # Compute GCUs as the ratio of used volume to total tote volume
        gcus = total_volumes / self.tote_volume
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
            env_ids = torch.arange(self.num_envs, device=self.device)

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
