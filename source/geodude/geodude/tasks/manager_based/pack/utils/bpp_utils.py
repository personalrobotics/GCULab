import hashlib
import math
import os
import pickle
import time
from collections import deque
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple

import isaaclab.utils.math as math_utils
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from packing3d import (
    Attitude,
    Container,
    Display,
    Item,
    PackingProblem,
    Position,
    Transform,
)
from geodude.tasks.manager_based.pack.utils.tote_helpers import (
    calculate_rotated_bounding_box_np,
)


# Define worker functions at module level for picklability
def create_items_worker(args):
    env_idx, obj_voxels, obj_asset_paths, objects = args
    items = []
    for j in objects:
        if j < len(obj_voxels) and obj_voxels[j] is not None:
            items.append(
                Item(
                    np.array(obj_voxels[j], dtype=np.float32), obj_asset_paths[j] if j < len(obj_asset_paths) else None
                )
            )
    return env_idx, items


def create_problem_worker(args):
    env_idx, tote_dims, items = args
    problem = PackingProblem(tote_dims, items)
    return env_idx, problem


class BPP:
    """
    Bin Packing Problem (BPP) utility class for managing 3D packing operations
    across multiple environments with multiprocessing support.
    """

    # Class constants
    MAX_SOURCE_EJECT_TRIES = 6
    MAX_WORKERS = 25
    UNUSED_VOLUME_BUFFER = 5000  # 5L buffer for unpackable volume
    GRID_SEARCH_NUM = 25
    STEP_WIDTH = 90

    # Cache directory for storing precomputed packing components
    CACHE_DIR = os.path.join(os.path.expanduser("~"), ".prl_cp_cache", "packing_cache")

    def __init__(self, tote_manager, num_envs: int, objects: list, scale: float = 1.0, **kwargs):
        """
        Initialize the BPP utility class.

        Args:
            tote_manager: Manager for tote operations
            num_envs: Number of environments
            objects: List of objects to pack
            scale: Scale factor for dimensions
            **kwargs: Additional configuration options
        """
        self.scale = scale
        self.objects = objects
        self.tote_manager = tote_manager
        self.num_envs = num_envs
        self.kwargs = kwargs

        # Create cache directory if it doesn't exist
        os.makedirs(self.CACHE_DIR, exist_ok=True)

        # Initialize environment-specific tracking
        self.packed_obj_idx = [[] for _ in range(num_envs)]
        self.unpackable_obj_idx = [[] for _ in range(num_envs)]
        self.source_eject_tries = [0] * num_envs
        self.curr_obj_indices = [None] * num_envs
        self.source_loaded = [False] * num_envs
        self.subset_obj_indices = {}

        self.fifo_queues = [deque() for _ in range(num_envs)]

        # Initialize unique properties storage
        self.unique_obj_dims = {}
        self.unique_obj_voxels = {}
        self.asset_path_to_id = {}
        self.id_to_asset_path = {}

        # Cache for efficient asset ID lookups
        self._obj_to_asset_id_cache = None

        # Setup packing problem components
        self._initialize_packing_components()

    def _get_cache_key(self):
        """Generate a unique cache key based on environment configuration"""
        # Get seed from environment if available
        seed = self.tote_manager.env.unwrapped.cfg.seed
        # Create a hash of the key components
        key_components = [
            f"seed={seed}",
            f"num_envs={self.num_envs}",
            f"num_objects={len(self.objects)}",
            f"scale={self.scale}",
        ]

        # Create a hash of all components
        key_str = "_".join(key_components)
        print("Generating cache key:", key_str)
        return hashlib.md5(key_str.encode()).hexdigest()

    def _initialize_packing_components(self):
        """Initialize tote dimensions, object data, and packing problems."""
        start_time = time.time()
        print("Initializing packing components...")

        # Check if we have a cached version
        cache_key = self._get_cache_key()
        cache_file = os.path.join(self.CACHE_DIR, f"{cache_key}.pkl")

        if os.path.exists(cache_file):
            print(f"Loading cached packing components from {cache_file}")
            try:
                with open(cache_file, "rb") as f:
                    cached_data = pickle.load(f)
                    self.tote_dims = cached_data["tote_dims"]
                    self.problems = cached_data["problems"]
                    self.unique_obj_dims = cached_data["unique_obj_dims"]
                    self.unique_obj_voxels = cached_data["unique_obj_voxels"]
                    self.unique_obj_latents = cached_data["unique_obj_latents"]
                    self.asset_path_to_id = cached_data["asset_path_to_id"]
                    self.id_to_asset_path = cached_data["id_to_asset_path"]
                    # Build cache after loading
                    self._build_obj_to_asset_id_cache()
                print(f"Successfully loaded cached data in {time.time() - start_time:.3f}s")
                return
            except Exception as e:
                print(f"Error loading cache: {e}. Rebuilding components...")

        # If we reach here, we need to build the components from scratch
        self.tote_dims, _, _, _ = self._get_packing_variables()

        # Convert all data to CPU and NumPy before multiprocessing to avoid CUDA errors
        cpu_obj_voxels = []
        cpu_asset_paths = []

        for i in range(self.num_envs):
            # Convert voxels to CPU/NumPy if they're tensors using unique properties
            env_voxels = []
            for j in self.objects:
                asset_path = self.tote_manager.obj_asset_paths[i][j]
                asset_id = self.asset_path_to_id.get(asset_path, None)
                voxel = self.unique_obj_voxels.get(asset_id, None) if asset_id is not None else None
                if voxel is not None and isinstance(voxel, torch.Tensor):
                    voxel = voxel.cpu().numpy()
                env_voxels.append(voxel)
            cpu_obj_voxels.append(env_voxels)

            # Convert asset paths to CPU if needed
            env_paths = []
            for j in self.objects:
                path = self.tote_manager.obj_asset_paths[i][j]
                if isinstance(path, torch.Tensor):
                    path = path.cpu().numpy()
                env_paths.append(path)
            cpu_asset_paths.append(env_paths)

        # Prepare arguments for item creation with CPU data
        item_args = []
        for i in range(self.num_envs):
            # Ensure objects list is CPU-based
            objects_cpu = [obj if not isinstance(obj, torch.Tensor) else obj.cpu().item() for obj in self.objects]
            item_args.append((i, cpu_obj_voxels[i], cpu_asset_paths[i], objects_cpu))
        print(f"Time to prepare item arguments: {time.time() - start_time:.3f}s")

        # Create items using multiprocessing with limited workers to avoid memory issues
        all_items = [None] * self.num_envs
        with ProcessPoolExecutor(max_workers=min(self.MAX_WORKERS, 4)) as executor:
            for env_idx, items in executor.map(create_items_worker, item_args):
                all_items[env_idx] = items
        print(f"Time to create items: {time.time() - start_time:.3f}s")

        # Prepare arguments for problem creation (tote_dims is already CPU/NumPy from _get_packing_variables)
        problem_args = [(i, self.tote_dims, all_items[i]) for i in range(self.num_envs)]

        # Create problems using multiprocessing with limited workers
        self.problems = [None] * self.num_envs
        with ProcessPoolExecutor(max_workers=min(self.MAX_WORKERS, 4)) as executor:
            for env_idx, problem in executor.map(create_problem_worker, problem_args):
                self.problems[env_idx] = problem
        print(f"Time to create problems: {time.time() - start_time:.3f}s")

        # Save the components to cache for future use
        try:
            cache_data = {
                "tote_dims": self.tote_dims,
                "problems": self.problems,
                "unique_obj_dims": self.unique_obj_dims,
                "unique_obj_voxels": self.unique_obj_voxels,
                "unique_obj_latents": self.unique_obj_latents,
                "asset_path_to_id": self.asset_path_to_id,
                "id_to_asset_path": self.id_to_asset_path,
            }
            with open(cache_file, "wb") as f:
                pickle.dump(cache_data, f)
            print(f"Saved packing components to cache: {cache_file}")
        except Exception as e:
            print(f"Error saving to cache: {e}")

        print(f"Packing components initialization time: {time.time() - start_time:.3f}s")

    def _build_obj_to_asset_id_cache(self):
        """Build a cache for fast object ID to asset ID lookups."""
        if self._obj_to_asset_id_cache is not None:
            return

        self._obj_to_asset_id_cache = {}
        for env_idx in range(self.tote_manager.num_envs):
            self._obj_to_asset_id_cache[env_idx] = {}
            for obj_idx in range(self.tote_manager.num_objects):
                asset_path = self.tote_manager.obj_asset_paths[env_idx][obj_idx]
                asset_id = self.asset_path_to_id.get(asset_path, 0)
                self._obj_to_asset_id_cache[env_idx][obj_idx] = asset_id

    def get_asset_id(self, env_idx: int, obj_idx: int) -> int:
        """Get asset ID for a specific object efficiently."""
        if self._obj_to_asset_id_cache is None:
            self._build_obj_to_asset_id_cache()
        return self._obj_to_asset_id_cache[env_idx].get(obj_idx, 0)

    def get_asset_ids_batch(self, env_indices: list, obj_indices: list) -> torch.Tensor:
        """Get asset IDs for a batch of environments and objects efficiently."""
        if self._obj_to_asset_id_cache is None:
            self._build_obj_to_asset_id_cache()

        asset_ids = []
        for env_idx, obj_idx in zip(env_indices, obj_indices):
            asset_id = self._obj_to_asset_id_cache[env_idx].get(obj_idx, 0)
            asset_ids.append(asset_id)

        return torch.tensor(asset_ids, device=self.tote_manager.device, dtype=torch.float32)

    def _get_packing_variables(self) -> tuple[list[int], list, list]:
        """
        Extract and format packing variables from the tote manager.

        Returns:
            Tuple of (tote_dimensions, object_dimensions, object_voxels)

        Raises:
            ValueError: If scaling is applied inconsistently
        """
        # Convert tote dimensions from xyz to zxy format and scale
        tote_dims = self.tote_manager.true_tote_dim.tolist()
        tote_dims = [int(tote_dims[2] * self.scale), int(tote_dims[0] * self.scale), int(tote_dims[1] * self.scale)]

        # Get unique object dimensions and voxels by asset path ID
        self.unique_obj_dims = {}
        self.unique_obj_voxels = {}
        self.unique_obj_latents = {}
        self.asset_path_to_id = {}
        self.id_to_asset_path = {}

        # Extract unique asset paths and their properties, assign IDs
        asset_id = 0
        for asset_path, (volume, bbox, voxels, latents) in self.tote_manager.unique_object_properties.items():
            self.asset_path_to_id[asset_path] = asset_id
            self.id_to_asset_path[asset_id] = asset_path
            self.unique_obj_dims[asset_id] = (bbox * self.scale).to(dtype=torch.int32).tolist()
            self.unique_obj_voxels[asset_id] = voxels
            self.unique_obj_latents[asset_id] = latents
            asset_id += 1

        # No need to create full environment-specific arrays - use unique properties directly

        if self.scale != 1.0:
            raise ValueError(
                "Scaling applied to tote dimensions but not to voxel grid. This may lead to packing inaccuracies."
            )

        return tote_dims, None, None, None  # Return None since we're using unique properties directly

    @staticmethod
    def _update_container_worker(args: dict[str, Any]) -> tuple[int, list, list, Container]:
        """
        Static worker for multiprocessing container updates.

        Args:
            args: Dictionary containing environment data and processing parameters

        Returns:
            Tuple of (env_idx, results, container_items, container)
        """
        env_idx = args["env_idx"]
        tote_id = args["tote_id"]
        objects = args["objects"]
        problem = args.get("problem")
        items = args.get("items")

        results = []
        container_items = []

        if tote_id < 0:
            return (env_idx, results, container_items, None)

        if problem is not None and items is not None:
            problem.container.clear()

        for obj in objects:
            transform = BPP._calculate_object_transform(obj)
            results.append((obj["obj_idx"], transform))

            if problem is not None and items is not None:
                curr_item = items[obj["obj_idx"]]
                curr_item.transform(transform)
                problem.container.add_item(curr_item)
                container_items.append((obj["obj_idx"], transform))

        return (env_idx, results, container_items, problem.container if problem else None)

    @staticmethod
    def _calculate_object_transform(obj: dict[str, Any]) -> Transform:
        """Calculate the transform for an object based on its current state."""
        # Convert to tensors for mathematical operations
        asset_pos = torch.tensor(obj["asset_pos"])
        asset_quat = torch.tensor(obj["asset_quat"])
        bbox_offset = torch.tensor(obj["bbox_offset"])
        true_tote_dim = torch.tensor(obj["true_tote_dim"])
        tote_assets_state = torch.tensor(obj["tote_assets_state"])

        # Apply quaternion transformations
        quat_init = torch.tensor([1, 0, 0, 0])
        asset_quat = math_utils.quat_mul(math_utils.quat_inv(quat_init), asset_quat)

        # Calculate rotated bounding box
        rotated_half_dim = (
            calculate_rotated_bounding_box_np(bbox_offset.unsqueeze(0), asset_quat.unsqueeze(0), device="cpu") / 2.0
        ).squeeze(0)

        # Transform to tote coordinate system
        asset_pos = asset_pos - rotated_half_dim
        asset_pos[0] += true_tote_dim[0] / 2 * 0.01
        asset_pos[1] += true_tote_dim[1] / 2 * 0.01
        asset_pos = asset_pos - tote_assets_state
        asset_pos = asset_pos * 100  # Convert to cm resolution

        # Convert quaternion to euler angles
        euler_angles = math_utils.euler_xyz_from_quat(asset_quat.unsqueeze(0))
        if isinstance(euler_angles, tuple):
            euler_angles = torch.tensor(euler_angles)
        euler_angles = euler_angles.squeeze(0) * 180 / np.pi

        return Transform(
            position=Position(
                max(0, math.floor(asset_pos[0].item())),
                max(0, math.floor(asset_pos[1].item())),
                max(0, math.floor(asset_pos[2].item())),
            ),
            attitude=Attitude(roll=euler_angles[0].item(), pitch=euler_angles[1].item(), yaw=euler_angles[2].item()),
        )

    def _extract_env_data(self, env, env_idx: int, tote_id: int) -> list[dict[str, Any]]:
        """
        Extract all necessary data from environment for a single environment index.

        Args:
            env: The environment object
            env_idx: Environment index
            tote_id: Tote identifier

        Returns:
            List of dictionaries containing object data
        """
        data = []
        for obj_idx in self.packed_obj_idx[env_idx]:
            obj_idx_val = int(obj_idx.item())
            asset_name = f"object{obj_idx_val}"
            asset = env.unwrapped.scene[asset_name]

            data.append({
                "obj_idx": obj_idx_val,
                "asset_pos": asset.data.root_state_w[env_idx, :3].detach().cpu().numpy(),
                "asset_quat": asset.data.root_state_w[env_idx, 3:7].detach().cpu().numpy(),
                "bbox_offset": self.tote_manager.get_object_bbox(env_idx, obj_idx_val).detach().cpu().numpy(),
                "true_tote_dim": self.tote_manager.true_tote_dim.detach().cpu().numpy(),
                "tote_assets_state": (
                    self.tote_manager._tote_assets_state.permute(1, 0, 2).detach().cpu().numpy()[env_idx, tote_id]
                ),
            })
        return data

    def update_container_heightmap(self, env, env_indices: torch.Tensor, tote_ids: torch.Tensor):
        """
        Update container heightmaps using batched multiprocessing.

        Args:
            env: Environment object
            env_indices: Tensor of environment indices
            tote_ids: Tensor of tote IDs
        """
        start_time = time.time()

        # Prepare batch arguments for multiprocessing
        batch_args = []
        for env_idx in env_indices:
            tote_id = tote_ids[env_idx].item()
            if tote_id < 0:
                continue

            objects = self._extract_env_data(env, env_idx, tote_id)
            batch_args.append({
                "env_idx": env_idx.detach().cpu().item(),
                "tote_id": tote_id,
                "objects": objects,
                "problem": self.problems[env_idx],
                "items": self.problems[env_idx].items,
            })

        # # Execute multiprocessing
        with ProcessPoolExecutor(max_workers=self.MAX_WORKERS) as executor:
            results = list(executor.map(self._update_container_worker, batch_args))

        # Update problems with results
        for env_idx, res, container_items, container in results:
            if container is not None:
                self.problems[env_idx].container = container
                self.tote_manager.stats.log_container(env_idx, container)

        print(f"Container heightmap update time: {time.time() - start_time:.3f}s")

    @staticmethod
    def _subset_sum_bfs(nums: list[float], idxs: list[int], max_volume: float) -> list[int]:
        """
        Find the best subset of objects that fit within the maximum volume using BFS.

        Args:
            nums: List of object volumes
            idxs: Corresponding object indices
            max_volume: Maximum total volume constraint

        Returns:
            List of object indices forming the best subset
        """
        queue = deque([(0.0, 0, [])])  # (current_sum, index, subset_indices)
        best_sum = 0.0
        best_subset = []

        while queue:
            curr_sum, idx, subset = queue.popleft()

            if curr_sum > best_sum:
                best_sum = curr_sum
                best_subset = subset

            if idx >= len(nums):
                continue

            for j in range(idx, len(nums)):
                new_sum = curr_sum + nums[j]
                if new_sum <= max_volume:
                    queue.append((new_sum, j + 1, subset + [idxs[j]]))
        return best_subset

    @staticmethod
    def _env_worker(args: tuple) -> tuple[int, tuple]:
        """
        Worker function for processing individual environments.

        Args:
            args: Tuple containing all processing parameters

        Returns:
            Tuple of (environment_index, processing_results)
        """
        (
            i,
            env_idx,
            obj_indices,
            problem,
            obj_volumes,
            dest_gcu,
            use_stability,
            use_subset_sum,
            use_desc_vol,
            source_tote_ejected,
            subset_obj_indices,
            plot,
        ) = args

        if not obj_indices:
            return (i, (None, None, None, None))

        print(f"Processing environment {env_idx} with {len(obj_indices)} objects.")

        curr_obj_indices = BPP._determine_object_subset(
            obj_indices, obj_volumes, problem, use_subset_sum, source_tote_ejected, subset_obj_indices, env_idx
        )

        if use_desc_vol and len(curr_obj_indices) > 1:
            # Sort by volume in descending order
            obj_vols = [(idx, obj_volumes[idx].item()) for idx in curr_obj_indices]
            curr_obj_indices = [idx for idx, _ in sorted(obj_vols, key=lambda x: x[1], reverse=True)]
        else:
            # Use original order if not sorting by volume
            pass
            # Randomly shuffle objects
            # shuffled = torch.randperm(len(curr_obj_indices), device="cpu")
            # curr_obj_indices = [curr_obj_indices[i] for i in shuffled]

        if not curr_obj_indices:
            return (i, (None, None, None, curr_obj_indices))

        # Try to pack the first object
        item = problem.items[curr_obj_indices[0]]
        obj_idx = curr_obj_indices[0].item()

        # Remove from subset regardless of packing success
        if use_subset_sum and curr_obj_indices:
            curr_obj_indices = [idx for idx in curr_obj_indices if idx != obj_idx]

        # Search for valid transforms
        transforms = problem.container.search_possible_position(
            item, grid_num=BPP.GRID_SEARCH_NUM, step_width=BPP.STEP_WIDTH
        )

        if use_stability:
            for transform in transforms:
                test_item = deepcopy(item)
                test_item.transform(transform)
                if problem.container.check_stability_with_candidate(test_item, plot=plot):
                    return (i, (obj_idx, torch.tensor([obj_idx]), transform, curr_obj_indices))
        elif transforms:
            return (i, (obj_idx, torch.tensor([obj_idx]), transforms[0], curr_obj_indices))

        return (i, (None, torch.tensor([obj_idx]), None, curr_obj_indices))

    @staticmethod
    def _determine_object_subset(
        obj_indices: list,
        obj_volumes: torch.Tensor,
        problem: PackingProblem,
        use_subset_sum: bool,
        source_tote_ejected: bool,
        subset_obj_indices: list | None,
        env_idx: int,
    ) -> list:
        """Determine which objects to consider for packing based on configuration."""
        if not use_subset_sum:
            print("Use subset sum is disabled, using all objects.")
            return obj_indices

        if source_tote_ejected:
            # Calculate available volume based on container state
            container = problem.container
            heightmap = container.heightmap
            unused_height = np.clip(container.box_size[0] - heightmap, 0, None)
            unused_height_augmented = unused_height.copy()
            unused_height_augmented[unused_height < 5] = 0  # Ignore gaps less than 5cm
            max_volume = np.sum(unused_height_augmented) - BPP.UNUSED_VOLUME_BUFFER

            if max_volume < 0:
                print(f"Warning: No usable volume left for environment {env_idx}")
                return []

            volumes = [obj_volumes[idx].item() for idx in obj_indices]
            best_subset = BPP._subset_sum_bfs(volumes, obj_indices, max_volume)
            print(f"SUBSET: New subset indices for environment {env_idx}: {sorted(best_subset)}")
            return best_subset
        else:
            print(
                f"SUBSET: Current subset indices for environment {env_idx}: {sorted(subset_obj_indices or obj_indices)}"
            )
            return subset_obj_indices or obj_indices

    def _eject_and_reload(self, env_idx: int, tote_ids: torch.Tensor, is_dest: bool, overfill_check: bool = False):
        """
        Eject and reload totes for the specified environment.

        Args:
            env_idx: Environment index
            tote_ids: Tensor of tote IDs
            is_dest: Whether ejecting destination tote
            overfill_check: Whether to perform overfill check
        """
        if is_dest:
            tote_tensor = tote_ids[env_idx].unsqueeze(0)
        else:
            tote_idx = self.source_eject_tries[env_idx] % len(tote_ids[env_idx])
            tote_tensor = tote_ids[env_idx][tote_idx].unsqueeze(0)

            # Remove objects from unpackable list when ejecting source
            tote_objects = self.tote_manager.get_tote_objs_idx(
                tote_tensor, torch.tensor([env_idx], device=self.tote_manager.device)
            )
            objects_to_remove = torch.where(tote_objects == 1)[1]
            self.unpackable_obj_idx[env_idx] = [
                idx for idx in self.unpackable_obj_idx[env_idx] if idx not in objects_to_remove
            ]

        # Perform ejection
        self.tote_manager.eject_totes(
            tote_tensor,
            torch.tensor([env_idx], device=self.tote_manager.device),
            is_dest=is_dest,
            overfill_check=overfill_check,
        )

        if is_dest:
            # Reset environment state for destination ejection
            self.problems[env_idx].container = Container(self.tote_dims)
            self.packed_obj_idx[env_idx] = []
            self.unpackable_obj_idx[env_idx] = []
            self.source_eject_tries[env_idx] = 0
        else:
            # Refill source totes and increment try counter
            self.tote_manager.refill_source_totes(torch.tensor([env_idx], device=self.tote_manager.device))
            self.source_eject_tries[env_idx] += 1

    def _get_new_objects(self, env_idx: int, dest_tote_ids: torch.Tensor) -> list:
        """Get new packable objects for the environment."""
        new_packable_objects, _ = self.get_packable_object_indices(
            self.tote_manager.num_objects,
            self.tote_manager,
            torch.tensor([env_idx]),
            dest_tote_ids[env_idx].unsqueeze(0).cpu(),
        )
        return [new_packable_objects[0][i].cpu() for i in range(len(new_packable_objects[0]))]

    def get_action(
        self, env, obj_indices: list[list], dest_tote_ids: torch.Tensor, env_indices: torch.Tensor, plot: bool = False
    ) -> tuple[list, torch.Tensor | None]:
        """
        Get packing actions for multiple environments using parallel processing.

        Args:
            env: Environment object
            obj_indices: List of object indices per environment
            dest_tote_ids: Destination tote IDs
            env_indices: Environment indices to process
            plot: Whether to enable plotting

        Returns:
            Tuple of (transforms_list, object_indices_tensor)

        Raises:
            ValueError: If no valid transforms or inconsistent results
        """
        start_time = time.time()

        # Setup source tote IDs (all totes except destination)
        source_tote_ids = torch.arange(0, self.tote_manager.num_totes, device=env.unwrapped.device)
        source_tote_ids = source_tote_ids.repeat(self.num_envs, 1)
        source_tote_ids = source_tote_ids[source_tote_ids != dest_tote_ids.unsqueeze(1)]
        source_tote_ids = source_tote_ids.reshape(self.num_envs, -1)

        # Initialize result containers
        transforms_list = [None] * len(env_indices)
        obj_idx_list = [None] * len(env_indices)

        # Convert object indices to CPU
        obj_indices = [[obj_indices[i][j].cpu() for j in range(len(obj_indices[i]))] for i in range(len(env_indices))]

        # Extract configuration options
        use_stability = self.kwargs.get("use_stability", False)
        use_subset_sum = self.kwargs.get("use_subset_sum", False)
        use_desc_vol = self.kwargs.get("decreasing_vol", False)

        # Process environments with parallel execution
        with ProcessPoolExecutor(max_workers=self.MAX_WORKERS) as executor:
            futures = {}

            # Submit initial jobs
            for i, env_idx in enumerate(env_indices.cpu()):
                curr_obj_indices = obj_indices[env_idx]
                job_args = self._prepare_job_args(
                    i, env_idx, curr_obj_indices, use_stability, use_subset_sum, use_desc_vol, plot, dest_tote_ids
                )
                future = executor.submit(self._env_worker, job_args)
                futures[future] = (i, env_idx, curr_obj_indices)

            # Process completed jobs and handle fallbacks
            while futures:
                for future in as_completed(futures):
                    result = self._process_completed_job(
                        future,
                        futures,
                        executor,
                        env,
                        env_indices,
                        dest_tote_ids,
                        source_tote_ids,
                        transforms_list,
                        obj_idx_list,
                        use_stability,
                        use_subset_sum,
                        use_desc_vol,
                        plot,
                        start_time,
                    )
                    if result:
                        break

        # Validate and return results
        valid_transforms = [t for t in transforms_list if t is not None]
        valid_obj_idxs = [o for o in obj_idx_list if o is not None]

        if not valid_transforms:
            raise ValueError("No valid transforms found for any environment.")
        if len(valid_obj_idxs) != self.num_envs:
            raise ValueError("Not all environments have valid object indices.")

        return valid_transforms, torch.cat(valid_obj_idxs, dim=0) if valid_obj_idxs else None

    def _prepare_job_args(
        self,
        i: int,
        env_idx: int,
        curr_obj_indices: list,
        use_stability: bool,
        use_subset_sum: bool,
        use_desc_vol: bool,
        plot: bool,
        dest_tote_ids: torch.Tensor = None,
    ) -> tuple:
        """Prepare arguments for worker job submission."""
        obj_volumes = torch.tensor(
            [self.tote_manager.get_object_volume(env_idx, obj_idx) for obj_idx in range(self.tote_manager.num_objects)]
        ).cpu()

        # Get GCU and ensure it's a scalar value - need to index by destination tote
        if dest_tote_ids is not None:
            gcu_tensor = self.tote_manager.get_gcu(torch.tensor([env_idx], device=self.tote_manager.device)).squeeze(0)
            dest_gcu = gcu_tensor[dest_tote_ids[env_idx]].item()
        else:
            # Fallback if dest_tote_ids not provided
            dest_gcu = 0.0
        subset_obj_indices = self.subset_obj_indices.get(env_idx.item(), None)

        return (
            i,
            env_idx,
            curr_obj_indices,
            self.problems[env_idx],
            obj_volumes,
            dest_gcu,
            use_stability,
            use_subset_sum,
            use_desc_vol,
            self.tote_manager.source_tote_ejected[env_idx],
            subset_obj_indices,
            plot,
        )

    def _process_completed_job(
        self,
        future,
        futures: dict,
        executor,
        env,
        env_indices: torch.Tensor,
        dest_tote_ids: torch.Tensor,
        source_tote_ids: torch.Tensor,
        transforms_list: list,
        obj_idx_list: list,
        use_stability: bool,
        use_subset_sum: bool,
        use_desc_vol: bool,
        plot: bool,
        start_time: float,
    ) -> bool:
        """Process a completed job and handle various fallback scenarios."""
        i, env_idx, curr_obj_indices = futures.pop(future)
        env_idx = torch.tensor(env_idx).to(env.unwrapped.device)

        _, (obj_idx, obj_idx_tensor, transform, subset_obj_indices) = future.result()

        # Update subset indices if using subset sum
        if subset_obj_indices is not None and use_subset_sum:
            self.subset_obj_indices[env_idx.item()] = subset_obj_indices
            curr_obj_indices = subset_obj_indices
        else:
            curr_obj_indices = [obj for obj in curr_obj_indices if obj not in self.unpackable_obj_idx[env_idx]]

        # Handle various fallback scenarios
        if self._should_eject_destination(env_idx):
            self._handle_destination_ejection(
                env_idx,
                dest_tote_ids,
                executor,
                futures,
                i,
                use_stability,
                use_subset_sum,
                use_desc_vol,
                subset_obj_indices,
                plot,
            )
        elif not curr_obj_indices:
            self._handle_empty_objects(
                env_idx,
                source_tote_ids,
                dest_tote_ids,
                executor,
                futures,
                i,
                use_stability,
                use_subset_sum,
                use_desc_vol,
                subset_obj_indices,
                plot,
            )
        elif obj_idx is not None:
            self._handle_successful_packing(
                i, env_idx, obj_idx, obj_idx_tensor, transform, transforms_list, obj_idx_list, start_time
            )
        elif obj_idx_tensor is not None:
            self._handle_unpackable_object(
                env_idx,
                obj_idx_tensor,
                curr_obj_indices,
                executor,
                futures,
                i,
                use_stability,
                use_subset_sum,
                use_desc_vol,
                subset_obj_indices,
                plot,
                dest_tote_ids,
            )
        else:
            raise ValueError(f"No valid object index or transform found for environment {env_idx}")

        return False  # Continue processing

    def _should_eject_destination(self, env_idx: int) -> bool:
        """Check if destination tote should be ejected."""
        return (
            self.source_eject_tries[env_idx] >= self.MAX_SOURCE_EJECT_TRIES
            or self.tote_manager.get_reserved_objs_idx(torch.tensor([env_idx], device=self.tote_manager.device)).sum()
            == 0
        )

    def _handle_destination_ejection(
        self,
        env_idx: int,
        dest_tote_ids: torch.Tensor,
        executor,
        futures: dict,
        i: int,
        use_stability: bool,
        use_subset_sum: bool,
        use_desc_vol: bool,
        subset_obj_indices: list | None,
        plot: bool,
    ):
        """Handle destination tote ejection scenario."""
        reason = (
            "Max source eject tries reached"
            if self.source_eject_tries[env_idx] >= self.MAX_SOURCE_EJECT_TRIES
            else "No reserved objects"
        )
        print(f"{reason} for environment {env_idx}. Ejecting destination tote.")

        self._eject_and_reload(env_idx, dest_tote_ids, is_dest=True)
        new_objs = self._get_new_objects(env_idx, dest_tote_ids)

        job_args = self._prepare_job_args(
            i, env_idx.cpu(), new_objs, use_stability, use_subset_sum, use_desc_vol, plot, dest_tote_ids
        )
        future = executor.submit(self._env_worker, job_args)
        futures[future] = (i, env_idx.cpu(), new_objs)

    def _handle_empty_objects(
        self,
        env_idx: int,
        source_tote_ids: torch.Tensor,
        dest_tote_ids: torch.Tensor,
        executor,
        futures: dict,
        i: int,
        use_stability: bool,
        use_subset_sum: bool,
        use_desc_vol: bool,
        subset_obj_indices: list | None,
        plot: bool,
    ):
        """Handle scenario where no packable objects remain."""
        print(f"No packable objects left in environment {env_idx}. Ejecting source tote.")

        self._eject_and_reload(env_idx, source_tote_ids, is_dest=False)
        new_objs = self._get_new_objects(env_idx, dest_tote_ids)
        print(f"New packable objects for environment {env_idx}: {new_objs}")

        job_args = self._prepare_job_args(
            i, env_idx.cpu(), new_objs, use_stability, use_subset_sum, use_desc_vol, plot, dest_tote_ids
        )
        future = executor.submit(self._env_worker, job_args)
        futures[future] = (i, env_idx.cpu(), new_objs)

    def _handle_successful_packing(
        self,
        i: int,
        env_idx: int,
        obj_idx: int,
        obj_idx_tensor: torch.Tensor,
        transform: Transform,
        transforms_list: list,
        obj_idx_list: list,
        start_time: float,
    ):
        """Handle successful object packing."""
        transforms_list[i] = transform
        obj_idx_list[i] = obj_idx_tensor
        self.packed_obj_idx[env_idx].append(obj_idx_tensor)
        print(f"Placement time for environment {env_idx}: {time.time() - start_time:.3f}s")

    def _handle_unpackable_object(
        self,
        env_idx: int,
        obj_idx_tensor: torch.Tensor,
        curr_obj_indices: list,
        executor,
        futures: dict,
        i: int,
        use_stability: bool,
        use_subset_sum: bool,
        use_desc_vol: bool,
        subset_obj_indices: list | None,
        plot: bool,
        dest_tote_ids: torch.Tensor,
    ):
        """Handle objects that cannot be packed."""
        print(f"No valid transform for object {obj_idx_tensor} in environment {env_idx}. Adding to unpackable objects.")

        self.unpackable_obj_idx[env_idx].extend(obj_idx_tensor)
        remaining_objs = [obj for obj in curr_obj_indices if obj not in self.unpackable_obj_idx[env_idx]]

        job_args = self._prepare_job_args(
            i, env_idx.cpu(), remaining_objs, use_stability, use_subset_sum, use_desc_vol, plot, dest_tote_ids
        )
        future = executor.submit(self._env_worker, job_args)
        futures[future] = (i, env_idx.cpu(), remaining_objs)

    def get_packable_object_indices(
        self, num_obj_per_env: int, tote_manager, env_indices: torch.Tensor, tote_ids: torch.Tensor
    ) -> tuple[list, torch.Tensor]:
        """
        Get indices of objects that can be packed per environment.

        Args:
            num_obj_per_env: Number of objects per environment
            tote_manager: The tote manager object
            env_indices: Indices of environments to get packable objects for
            tote_ids: Destination tote IDs for each environment

        Returns:
            Tuple of (list of packable object indices per environment, mask tensor)
        """
        num_envs = env_indices.shape[0]

        # Get objects that are reserved (already being picked up)
        reserved_objs = tote_manager.get_reserved_objs_idx(env_indices)

        # Get objects that are already in destination totes
        objs_in_dest = tote_manager.get_tote_objs_idx(tote_ids, env_indices)

        # Create a 2D tensor of object indices: shape (num_envs, num_obj_per_env)
        obj_indices = torch.arange(0, num_obj_per_env, device="cpu").expand(num_envs, -1)

        # Compute mask of packable objects
        mask = (~reserved_objs & ~objs_in_dest).bool().cpu()

        # Get valid indices per environment
        valid_indices = [obj_indices[i][mask[i]] for i in range(num_envs)]

        # Remove unpackable objects from valid indices
        for i in range(num_envs):
            env_idx = env_indices[i].item()
            valid_indices[i] = [obj for obj in valid_indices[i] if obj not in self.unpackable_obj_idx[env_idx]]

        return valid_indices, mask

    def select_fifo_packable_objects(self, packable_objects, device):
        """
        Select packable objects using FIFO (First In, First Out) ordering.

        Args:
            packable_objects: List of tensors with packable object indices for each environment
            device: Device to create tensors on

        Returns:
            Tensor of selected object indices (-1 for environments with no packable objects)
        """
        import torch

        num_envs = len(packable_objects)
        selected_obj_indices = torch.full((num_envs,), -1, device=device, dtype=torch.int32)

        for env_idx, packable_list in enumerate(packable_objects):
            if len(packable_list) == 0:
                continue

            packable_values = {obj.item() for obj in packable_list}

            # Remove stale objects from front of FIFO
            while self.fifo_queues[env_idx] and self.fifo_queues[env_idx][0].item() not in packable_values:
                self.fifo_queues[env_idx].popleft()

            # Pick the first object from FIFO, but don't remove it yet
            if self.fifo_queues[env_idx]:
                selected_obj_indices[env_idx] = self.fifo_queues[env_idx][0]  # Peek at first object
            else:
                # If FIFO is empty, pick the first available packable object
                selected_obj_indices[env_idx] = packable_list[0]

        return selected_obj_indices

    def update_fifo_queues(self, packable_objects):
        """
        Update FIFO queues with new packable objects while maintaining FIFO order.

        Args:
            packable_objects: List of tensors with packable object indices for each environment
        """
        for env_idx, packable_list in enumerate(packable_objects):
            fifo_queue = self.fifo_queues[env_idx]
            packable_values = {obj.item() for obj in packable_list}

            # Remove stale objects from FIFO that are no longer packable
            from collections import deque

            self.fifo_queues[env_idx] = deque([obj for obj in fifo_queue if obj.item() in packable_values])

            # Append new objects that aren't already in FIFO
            fifo_values = {obj.item() for obj in self.fifo_queues[env_idx]}
            for obj in packable_list:
                if obj.item() not in fifo_values:
                    self.fifo_queues[env_idx].append(obj)
                    fifo_values.add(obj.item())

            # REORDER packable_objects to match FIFO order
            packable_objects[env_idx] = list(self.fifo_queues[env_idx])

    def remove_selected_from_fifo(self, selected_objects):
        """
        Remove selected objects from the front of FIFO queues.

        Args:
            selected_objects: Tensor of selected object indices
        """
        for env_idx, selected_obj in enumerate(selected_objects):
            if (
                selected_obj != -1
                and self.fifo_queues[env_idx]
                and self.fifo_queues[env_idx][0].item() == selected_obj.item()
            ):
                self.fifo_queues[env_idx].popleft()
