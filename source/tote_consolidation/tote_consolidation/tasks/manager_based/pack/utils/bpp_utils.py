import math
import time
from collections import deque
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
from typing import List, Tuple, Optional, Dict, Any

import isaaclab.utils.math as math_utils
import matplotlib
matplotlib.use("Agg")  # Use non-interactive backend for saving figures
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
from tote_consolidation.tasks.manager_based.pack.utils.tote_helpers import (
    calculate_rotated_bounding_box_np,
)


class BPP:
    """
    Bin Packing Problem (BPP) utility class for managing 3D packing operations
    across multiple environments with multiprocessing support.
    """
    
    # Class constants
    MAX_SOURCE_EJECT_TRIES = 6
    MAX_WORKERS = 20
    UNUSED_VOLUME_BUFFER = 5000  # 5L buffer for unpackable volume
    GRID_SEARCH_NUM = 25
    STEP_WIDTH = 90
    
    def __init__(self, tote_manager, num_envs: int, objects: List, scale: float = 1.0, **kwargs):
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
        
        # Initialize environment-specific tracking
        self.packed_obj_idx = [[] for _ in range(num_envs)]
        self.unpackable_obj_idx = [[] for _ in range(num_envs)]
        self.source_eject_tries = [0] * num_envs
        self.curr_obj_indices = [None] * num_envs
        self.source_loaded = [False] * num_envs
        self.subset_obj_indices = {}
        
        # Setup packing problem components
        self._initialize_packing_components()

    def _initialize_packing_components(self):
        """Initialize tote dimensions, object data, and packing problems."""
        self.tote_dims, self.obj_dims, self.obj_voxels = self._get_packing_variables()
        
        # Create items for each environment
        items = [
            [
                Item(
                    np.array(self.obj_voxels[i][j], dtype=np.float32), 
                    self.tote_manager.obj_asset_paths[i][j]
                )
                for j in self.objects
            ]
            for i in range(self.num_envs)
        ]
        
        self.display = Display(self.tote_dims)
        self.problems = [
            PackingProblem(self.tote_dims, items[i]) 
            for i in range(self.num_envs)
        ]

    def _get_packing_variables(self) -> Tuple[List[int], List, List]:
        """
        Extract and format packing variables from the tote manager.
        
        Returns:
            Tuple of (tote_dimensions, object_dimensions, object_voxels)
            
        Raises:
            ValueError: If scaling is applied inconsistently
        """
        # Convert tote dimensions from xyz to zxy format and scale
        tote_dims = self.tote_manager.true_tote_dim.tolist()
        tote_dims = [
            int(tote_dims[2] * self.scale),
            int(tote_dims[0] * self.scale), 
            int(tote_dims[1] * self.scale)
        ]

        # Scale object bounding boxes
        obj_dims = (self.tote_manager.obj_bboxes * self.scale).to(dtype=torch.int32).tolist()
        obj_voxels = self.tote_manager.obj_voxels
        
        if self.scale != 1.0:
            raise ValueError(
                "Scaling applied to tote dimensions but not to voxel grid. "
                "This may lead to packing inaccuracies."
            )
            
        return tote_dims, obj_dims, obj_voxels

    @staticmethod
    def _update_container_worker(args: Dict[str, Any]) -> Tuple[int, List, List, Container]:
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
                
        return (env_idx, results, container_items, 
                problem.container if problem else None)

    @staticmethod
    def _calculate_object_transform(obj: Dict[str, Any]) -> Transform:
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
            calculate_rotated_bounding_box_np(
                bbox_offset.unsqueeze(0), 
                asset_quat.unsqueeze(0), 
                device="cpu"
            ) / 2.0
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
            attitude=Attitude(
                roll=euler_angles[0].item(),
                pitch=euler_angles[1].item(),
                yaw=euler_angles[2].item()
            ),
        )

    def _extract_env_data(self, env, env_idx: int, tote_id: int) -> List[Dict[str, Any]]:
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
                "bbox_offset": self.tote_manager.obj_bboxes[env_idx, obj_idx_val].detach().cpu().numpy(),
                "true_tote_dim": self.tote_manager.true_tote_dim.detach().cpu().numpy(),
                "tote_assets_state": self.tote_manager._tote_assets_state.permute(1, 0, 2).detach().cpu().numpy()[env_idx, tote_id],
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
        
        # results = [self._update_container_worker(args) for args in batch_args]
        
        # Update problems with results
        for env_idx, res, container_items, container in results:
            if container is not None:
                self.problems[env_idx].container = container
                self.tote_manager.stats.log_container(env_idx, container)
        
        print(f"Container heightmap update time: {time.time() - start_time:.3f}s")

    @staticmethod
    def _subset_sum_bfs(nums: List[float], idxs: List[int], max_volume: float) -> List[int]:
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
    def _env_worker(args: Tuple) -> Tuple[int, Tuple]:
        """
        Worker function for processing individual environments.
        
        Args:
            args: Tuple containing all processing parameters
            
        Returns:
            Tuple of (environment_index, processing_results)
        """
        (i, env_idx, obj_indices, problem, obj_volumes, dest_gcu, 
         use_stability, use_subset_sum, use_desc_vol, source_tote_ejected, 
         subset_obj_indices, plot) = args
        
        if not obj_indices:
            return (i, (None, None, None, None))

        print(f"Processing environment {env_idx} with {len(obj_indices)} objects.")
        
        curr_obj_indices = BPP._determine_object_subset(
            obj_indices, obj_volumes, problem, use_subset_sum, 
            source_tote_ejected, subset_obj_indices, env_idx
        )

        
        if use_desc_vol and len(curr_obj_indices) > 1:
            # Sort by volume in descending order
            obj_vols = [(idx, obj_volumes[idx].item()) for idx in curr_obj_indices]
            curr_obj_indices = [idx for idx, _ in sorted(obj_vols, key=lambda x: x[1], reverse=True)]
        else:
            # Randomly shuffle objects
            shuffled = torch.randperm(len(curr_obj_indices), device="cpu")
            curr_obj_indices = [curr_obj_indices[i] for i in shuffled]
        
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
            item, grid_num=BPP.GRID_SEARCH_NUM, 
            step_width=BPP.STEP_WIDTH
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
    def _determine_object_subset(obj_indices: List, obj_volumes: torch.Tensor, 
                               problem: PackingProblem, use_subset_sum: bool, 
                               source_tote_ejected: bool, subset_obj_indices: Optional[List],
                               env_idx: int) -> List:
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
            print(f"SUBSET: Current subset indices for environment {env_idx}: {sorted(subset_obj_indices or obj_indices)}")
            return subset_obj_indices or obj_indices

    def _eject_and_reload(self, env_idx: int, tote_ids: torch.Tensor, 
                         is_dest: bool, overfill_check: bool = False):
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
                idx for idx in self.unpackable_obj_idx[env_idx] 
                if idx not in objects_to_remove
            ]
        
        # Perform ejection
        self.tote_manager.eject_totes(
            tote_tensor, torch.tensor([env_idx], device=self.tote_manager.device), 
            is_dest=is_dest, overfill_check=overfill_check
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

    def _get_new_objects(self, env_idx: int, dest_tote_ids: torch.Tensor) -> List:
        """Get new packable objects for the environment."""
        new_packable_objects, _ = self.get_packable_object_indices(
            self.tote_manager.num_objects, self.tote_manager,
            torch.tensor([env_idx]), dest_tote_ids[env_idx].unsqueeze(0).cpu()
        )
        return [new_packable_objects[0][i].cpu() for i in range(len(new_packable_objects[0]))]

    def get_action(self, env, obj_indices: List[List], dest_tote_ids: torch.Tensor, 
                  env_indices: torch.Tensor, plot: bool = False) -> Tuple[List, Optional[torch.Tensor]]:
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
        obj_indices = [
            [obj_indices[i][j].cpu() for j in range(len(obj_indices[i]))] 
            for i in range(len(env_indices))
        ]
        
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
                    i, env_idx, curr_obj_indices, use_stability, 
                    use_subset_sum, use_desc_vol, plot, dest_tote_ids
                )
                future = executor.submit(self._env_worker, job_args)
                futures[future] = (i, env_idx, curr_obj_indices)
            
            # Process completed jobs and handle fallbacks
            while futures:
                for future in as_completed(futures):
                    result = self._process_completed_job(
                        future, futures, executor, env, env_indices, 
                        dest_tote_ids, source_tote_ids, transforms_list, 
                        obj_idx_list, use_stability, use_subset_sum, 
                        use_desc_vol, plot, start_time
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

    def _prepare_job_args(self, i: int, env_idx: int, curr_obj_indices: List,
                         use_stability: bool, use_subset_sum: bool, 
                         use_desc_vol: bool, plot: bool, dest_tote_ids: torch.Tensor = None) -> Tuple:
        """Prepare arguments for worker job submission."""
        obj_volumes = self.tote_manager.obj_volumes[env_idx].cpu()
        
        # Get GCU and ensure it's a scalar value - need to index by destination tote
        if dest_tote_ids is not None:
            gcu_tensor = self.tote_manager.get_gcu(torch.tensor([env_idx], device=self.tote_manager.device)).squeeze(0)
            dest_gcu = gcu_tensor[dest_tote_ids[env_idx]].item()
        else:
            # Fallback if dest_tote_ids not provided
            dest_gcu = 0.0
        subset_obj_indices = self.subset_obj_indices.get(env_idx.item(), None)
        
        return (
            i, env_idx, curr_obj_indices, self.problems[env_idx], obj_volumes,
            dest_gcu, use_stability, use_subset_sum, use_desc_vol,
            self.tote_manager.source_tote_ejected[env_idx], subset_obj_indices, plot
        )

    def _process_completed_job(self, future, futures: Dict, executor, env, 
                             env_indices: torch.Tensor, dest_tote_ids: torch.Tensor,
                             source_tote_ids: torch.Tensor, transforms_list: List,
                             obj_idx_list: List, use_stability: bool, 
                             use_subset_sum: bool, use_desc_vol: bool, 
                             plot: bool, start_time: float) -> bool:
        """Process a completed job and handle various fallback scenarios."""
        i, env_idx, curr_obj_indices = futures.pop(future)
        env_idx = torch.tensor(env_idx).to(env.unwrapped.device)
        
        _, (obj_idx, obj_idx_tensor, transform, subset_obj_indices) = future.result()
        
        # Update subset indices if using subset sum
        if subset_obj_indices is not None and use_subset_sum:
            self.subset_obj_indices[env_idx.item()] = subset_obj_indices
            curr_obj_indices = subset_obj_indices
        else:
            curr_obj_indices = [
                obj for obj in curr_obj_indices 
                if obj not in self.unpackable_obj_idx[env_idx]
            ]
        
        # Handle various fallback scenarios
        if self._should_eject_destination(env_idx):
            self._handle_destination_ejection(
                env_idx, dest_tote_ids, executor, futures, i, 
                use_stability, use_subset_sum, use_desc_vol, 
                subset_obj_indices, plot
            )
        elif not curr_obj_indices:
            self._handle_empty_objects(
                env_idx, source_tote_ids, dest_tote_ids, executor, 
                futures, i, use_stability, use_subset_sum, use_desc_vol, 
                subset_obj_indices, plot
            )
        elif obj_idx is not None:
            self._handle_successful_packing(
                i, env_idx, obj_idx, obj_idx_tensor, transform, 
                transforms_list, obj_idx_list, start_time
            )
        elif obj_idx_tensor is not None:
            self._handle_unpackable_object(
                env_idx, obj_idx_tensor, curr_obj_indices, executor, 
                futures, i, use_stability, use_subset_sum, use_desc_vol, 
                subset_obj_indices, plot, dest_tote_ids
            )
        else:
            raise ValueError(f"No valid object index or transform found for environment {env_idx}")
        
        return False  # Continue processing

    def _should_eject_destination(self, env_idx: int) -> bool:
        """Check if destination tote should be ejected."""
        return (
            self.source_eject_tries[env_idx] >= self.MAX_SOURCE_EJECT_TRIES or
            self.tote_manager.get_reserved_objs_idx(torch.tensor([env_idx], device=self.tote_manager.device)).sum() == 0
        )

    def _handle_destination_ejection(self, env_idx: int, dest_tote_ids: torch.Tensor,
                                   executor, futures: Dict, i: int, use_stability: bool,
                                   use_subset_sum: bool, use_desc_vol: bool,
                                   subset_obj_indices: Optional[List], plot: bool):
        """Handle destination tote ejection scenario."""
        reason = ("Max source eject tries reached" if 
                 self.source_eject_tries[env_idx] >= self.MAX_SOURCE_EJECT_TRIES
                 else "No reserved objects")
        print(f"{reason} for environment {env_idx}. Ejecting destination tote.")
        
        self._eject_and_reload(env_idx, dest_tote_ids, is_dest=True)
        new_objs = self._get_new_objects(env_idx, dest_tote_ids)
        
        job_args = self._prepare_job_args(
            i, env_idx.cpu(), new_objs, use_stability, 
            use_subset_sum, use_desc_vol, plot, dest_tote_ids
        )
        future = executor.submit(self._env_worker, job_args)
        futures[future] = (i, env_idx.cpu(), new_objs)

    def _handle_empty_objects(self, env_idx: int, source_tote_ids: torch.Tensor,
                            dest_tote_ids: torch.Tensor, executor, futures: Dict,
                            i: int, use_stability: bool, use_subset_sum: bool,
                            use_desc_vol: bool, subset_obj_indices: Optional[List], 
                            plot: bool):
        """Handle scenario where no packable objects remain."""
        print(f"No packable objects left in environment {env_idx}. Ejecting source tote.")
        
        self._eject_and_reload(env_idx, source_tote_ids, is_dest=False)
        new_objs = self._get_new_objects(env_idx, dest_tote_ids)
        print(f"New packable objects for environment {env_idx}: {new_objs}")
        
        job_args = self._prepare_job_args(
            i, env_idx.cpu(), new_objs, use_stability, 
            use_subset_sum, use_desc_vol, plot, dest_tote_ids
        )
        future = executor.submit(self._env_worker, job_args)
        futures[future] = (i, env_idx.cpu(), new_objs)

    def _handle_successful_packing(self, i: int, env_idx: int, obj_idx: int,
                                 obj_idx_tensor: torch.Tensor, transform: Transform,
                                 transforms_list: List, obj_idx_list: List, 
                                 start_time: float):
        """Handle successful object packing."""
        transforms_list[i] = transform
        obj_idx_list[i] = obj_idx_tensor
        self.packed_obj_idx[env_idx].append(obj_idx_tensor)
        print(f"Placement time for environment {env_idx}: {time.time() - start_time:.3f}s")

    def _handle_unpackable_object(self, env_idx: int, obj_idx_tensor: torch.Tensor,
                                curr_obj_indices: List, executor, futures: Dict,
                                i: int, use_stability: bool, use_subset_sum: bool,
                                use_desc_vol: bool, subset_obj_indices: Optional[List], 
                                plot: bool, dest_tote_ids: torch.Tensor):
        """Handle objects that cannot be packed."""
        print(f"No valid transform for object {obj_idx_tensor} in environment {env_idx}. "
              "Adding to unpackable objects.")
        
        self.unpackable_obj_idx[env_idx].extend(obj_idx_tensor)
        remaining_objs = [
            obj for obj in curr_obj_indices 
            if obj not in self.unpackable_obj_idx[env_idx]
        ]
        
        job_args = self._prepare_job_args(
            i, env_idx.cpu(), remaining_objs, use_stability, 
            use_subset_sum, use_desc_vol, plot, dest_tote_ids
        )
        future = executor.submit(self._env_worker, job_args)
        futures[future] = (i, env_idx.cpu(), remaining_objs)

    def get_packable_object_indices(self, num_obj_per_env: int, tote_manager, 
                                  env_indices: torch.Tensor, tote_ids: torch.Tensor) -> Tuple[List, torch.Tensor]:
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
            valid_indices[i] = [
                obj for obj in valid_indices[i] 
                if obj not in self.unpackable_obj_idx[env_idx]
            ]

        return valid_indices, mask