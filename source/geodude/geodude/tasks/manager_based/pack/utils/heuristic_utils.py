import math
import re
from typing import Any, Dict, List, Optional, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy

import numpy as np
import torch
from packing3d import Attitude, Container, Item, PackingProblem, Position, Transform

from geodude.tasks.manager_based.pack.utils import bpp_utils
num_placed_objects =0

# Worker function for creating items with YCB IDs (overrides bpp_utils version)
def create_items_worker(args):
    """Create items with obj_id set to YCB ID extracted from asset path."""
    env_idx, obj_voxels, obj_asset_paths, objects = args
    items = []
    for j in objects:
        if j < len(obj_voxels) and obj_voxels[j] is not None:
            # Extract YCB ID from asset path
            asset_path = obj_asset_paths[j] if j < len(obj_asset_paths) else None
            ycb_id = None
            if asset_path:
                ycb_id = int((re.search(r'/(\d{3})_', asset_path)).group(1))
            if ycb_id is None:
                raise ValueError(f"Could not extract YCB ID from asset path: {asset_path}")
            
            item = Item(
                np.array(obj_voxels[j], dtype=np.float32),
                obj_id=ycb_id
            )
            items.append(item)
    return env_idx, items


class ObjectType:
    """Object type classifications for heuristic packing."""
    CUBOIDAL = "cuboidal"
    CYLINDRICAL = "cylindrical"
    BOWL = "bowl"
    MUG = "mug"
    BANANA = "banana"
    IRREGULAR = "irregular"


class Heuristic(bpp_utils.BPP):
    """Heuristic-based packing agent using semantic masks and object-specific strategies."""
    
    # Zone definitions (x-coordinates in cm, container x_size is 50cm)
    CUBOIDAL_ZONE = (0, 30)  # Right 30 cm for cuboidals
    IRREGULAR_ZONE = (30, 50)  # left 20 cm for irregular objects
    
    # Object classification by YCB ID
    OBJECT_TYPE_MAP = {
        "003": ObjectType.CUBOIDAL,      # cracker_box
        "004": ObjectType.CUBOIDAL,      # sugar_box
        "010": ObjectType.CYLINDRICAL,   # potted_meat_can
        "011": ObjectType.BANANA,        # banana
        "024": ObjectType.BOWL,          # bowl
        "025": ObjectType.MUG,           # mug
    }

    def __init__(self, tote_manager, num_envs: int, objects: list, scale: float = 1.0, **kwargs):
        super().__init__(tote_manager, num_envs, objects, scale, **kwargs)
    
    def _get_items_worker(self):
        """Override to use heuristic_utils.create_items_worker with YCB IDs."""
        return create_items_worker
    
    def _get_cache_key(self):
        """Override to use separate cache for heuristic (items have YCB IDs instead of sequential)."""
        base_key = super()._get_cache_key()
        return base_key + "_heuristic"
    
    def get_semantic_mask(self, env_idx: int, as_tensor: bool = False, device: torch.device | None = None):
        """Get the top-down semantic mask for an environment."""
        mask = self.problems[env_idx].container.semantic_mask
        if as_tensor:
            device = device or self.tote_manager.device
            return torch.tensor(mask, device=device, dtype=torch.int32)
        return mask

    def classify_object_by_ycb_id(self, ycb_id: int) -> str:
        """Classify object type directly from YCB ID."""
        ycb_id_str = str(ycb_id).zfill(3)
        return self.OBJECT_TYPE_MAP.get(ycb_id_str, ObjectType.IRREGULAR)
    
    def classify_object(self, env_idx: int, obj_idx: int) -> str:
        """Classify object type based on YCB ID from item."""
        # Get the item's YCB ID (obj_id is now the YCB ID from initialization)
        item = self.problems[env_idx].items[obj_idx]
        if item.obj_id is None:
            return ObjectType.IRREGULAR
        
        return self.classify_object_by_ycb_id(item.obj_id)

    def get_target_zone(self, obj_type: str) -> Tuple[int, int]:
        """Get target x-coordinate zone for object type."""
        if obj_type in [ObjectType.CUBOIDAL, ObjectType.CYLINDRICAL]:
            return self.CUBOIDAL_ZONE
        return self.IRREGULAR_ZONE

    def find_stacking_candidates(
        self, container: Container, env_idx: int, obj_idx: int, obj_type: str
    ) -> List[Tuple[int, int, float]]:
        """Find candidate positions for stacking based on semantic mask.
        
        Returns list of (x, y, height) tuples where stacking is possible.
        """
        candidates = []
        semantic_mask = container.semantic_mask
        heightmap = container.heightmap
        
        # For each position in the mask
        for x in range(semantic_mask.shape[0]):
            for y in range(semantic_mask.shape[1]):
                existing_obj_id = semantic_mask[x, y]
                
                # Skip empty positions
                if existing_obj_id < 0:
                    continue
                    
                # Check if existing object is similar type (for stacking)
                # existing_obj_id in semantic mask is now a YCB ID
                existing_obj_type = self.classify_object_by_ycb_id(existing_obj_id)
                
                # Determine if stacking is suitable
                can_stack = False
                if obj_type == ObjectType.CUBOIDAL and existing_obj_type == ObjectType.CUBOIDAL:
                    can_stack = True
                elif obj_type == ObjectType.CYLINDRICAL and existing_obj_type == ObjectType.CYLINDRICAL:
                    can_stack = True
                elif obj_type == ObjectType.BOWL and existing_obj_type == ObjectType.BOWL:
                    can_stack = True  # Nesting
                elif obj_type == ObjectType.MUG and existing_obj_type == ObjectType.MUG:
                    can_stack = True
                    
                if can_stack:
                    height = heightmap[x, y]
                    candidates.append((x, y, height))
        
        # Sort by height (prefer stacking on taller objects for stability)
        candidates.sort(key=lambda c: c[2], reverse=True)
        return candidates

    @staticmethod
    def _generate_orientations_static(bbox: np.ndarray, obj_type: str) -> List[Attitude]:
        """Static method to generate candidate orientations based on object type."""
        # Sort dimensions to find longest, middle, shortest
        dims = sorted(enumerate(bbox), key=lambda x: x[1], reverse=True)
        longest_axis, middle_axis, shortest_axis = [d[0] for d in dims]
        
        orientations = []
        if obj_type == ObjectType.CUBOIDAL:
            # Stack vertically with longest side up
            if longest_axis == 0:  # x is longest
                orientations.append(Attitude(roll=0, pitch=90, yaw=0))
            elif longest_axis == 1:  # y is longest
                orientations.append(Attitude(roll=0, pitch=90, yaw=0))
            else:  # z is longest ("stands it up" and aligns it with shorter axis of tote)
                orientations.append(Attitude(roll=90, pitch=0, yaw=90))
            
            # Add flat orientations as backup
            # orientations.append(Attitude(roll=0, pitch=0, yaw=0))
            # orientations.append(Attitude(roll=90, pitch=0, yaw=0))
            # orientations.append(Attitude(roll=0, pitch=90, yaw=0))
            
        elif obj_type == ObjectType.CYLINDRICAL:
            # Cylinders: try vertical (standing) and horizontal (laying down)
            orientations.append(Attitude(roll=0, pitch=0, yaw=0))  # Standing
            orientations.append(Attitude(roll=90, pitch=0, yaw=0))  # Laying
            
        elif obj_type == ObjectType.BOWL:
            # Bowls: face up for nesting
            orientations.append(Attitude(roll=0, pitch=0, yaw=0))
            
        elif obj_type == ObjectType.MUG:
            # Mugs: face down for stacking
            orientations.append(Attitude(roll=180, pitch=0, yaw=0))
            orientations.append(Attitude(roll=0, pitch=0, yaw=0))  # Face up as backup
            
        elif obj_type == ObjectType.BANANA:
            # Bananas: lay flat
            orientations.append(Attitude(roll=90, pitch=0, yaw=0))
            orientations.append(Attitude(roll=0, pitch=90, yaw=0))
            
        else:  # IRREGULAR
            # Try common stable orientations
            orientations.extend([
                Attitude(roll=0, pitch=0, yaw=0),
                Attitude(roll=90, pitch=0, yaw=0),
                Attitude(roll=0, pitch=90, yaw=0),
            ])
        print(f" object of type {obj_type} has bbox {bbox} got orientations: {orientations}")
        return orientations

    @staticmethod
    def _heuristic_worker(args: dict) -> Tuple[int, Optional[int], Optional[Transform]]:
        """Worker function for finding object placement using heuristic strategy."""
        global num_placed_objects

        env_idx = args["env_idx"]
        obj_indices = args["obj_indices"]
        problem = args["problem"]
        obj_types = args["obj_types"]
        target_zones = args["target_zones"]
        stacking_candidates = args["stacking_candidates"]
        bbox_list = args["bbox_list"]
        
        if not obj_indices:
            return (env_idx, None, None)
        
        # Try each object in order
        for idx, obj_idx in enumerate(obj_indices):
            item = problem.items[obj_idx]
            obj_type = obj_types[idx]
            target_zone = target_zones[idx]
            stacking_cands = stacking_candidates[idx]
            bbox = bbox_list[idx]
            
            # Generate orientations on-demand
            orientations = Heuristic._generate_orientations_static(bbox, obj_type)
            
            # Try each orientation
            for attitude in orientations:
                item.rotate(attitude)
                item.calc_heightmap()

                # x = problem.container.geometry.x_size - 1
                # y = problem.container.geometry.y_size - 1   
                # while (not problem.container.add_item_topdown(item, x, y)):
                #     x -= 1
                #     y -= 1
                #     print(f"trying to add object to {x}, {y}")
                # if x < 0 or y < 0:
                #     print(f"failed to add object to bottom left?")
                #     return (env_idx, None, None)

                item.position = Position(num_placed_objects*8, 0, 0)
                
                    
                transform = Transform(item.position, attitude)
                print(f" object {obj_idx} in environment {env_idx} got transform: {transform}")
                print(f"container heightmap: {problem.container.heightmap}")
                print(f"container semantic mask: {problem.container.semantic_mask}")
                num_placed_objects += 1
                return (env_idx, obj_idx, transform)
                
                # # Priority 1: Try stacking on similar objects
                # for sx, sy, _ in stacking_cands:
                #     if problem.container.add_item_topdown(item, sx, sy):
                #         transform = Transform(item.position, attitude)
                #         return (env_idx, obj_idx, transform)
                
                # Priority 2: Try target zone with type-specific heuristic
                # zone_x_min, zone_x_max = target_zone
                # best_score = float('inf')
                # best_position = None
                
                # # Determine scoring heuristic based on object type
                # is_cuboidal = obj_type in [ObjectType.CUBOIDAL, ObjectType.CYLINDRICAL]
                
                # # Grid search in target zone
                # for x in range(zone_x_min, min(zone_x_max, problem.container.geometry.x_size)):
                #     for y in range(problem.container.geometry.y_size):
                #         if problem.container.add_item_topdown(item, x, y):
                #             if is_cuboidal:
                #                 # DBLF (Deep Bottom Left First): prioritize low z, then low x, then low y
                #                 score = item.position.z * 100 + x * 10 + y
                #             else:
                #                 # DTR (Deep Top Right): prioritize low z, then high x, then high y
                #                 # Negate x and y to maximize them while minimizing overall score
                #                 max_x = problem.container.geometry.x_size
                #                 max_y = problem.container.geometry.y_size
                #                 score = item.position.z * 100 + (max_x - x) * 10 + (max_y - y)
                            
                #             if score < best_score:
                #                 best_score = score
                #                 best_position = Transform(item.position, attitude)
                
                # if best_position is not None:
                #     return (env_idx, obj_idx, best_position)
                
                # # Priority 3: Try anywhere in container
                # for x in range(problem.container.geometry.x_size):
                #     for y in range(problem.container.geometry.y_size):
                #         if problem.container.add_item_topdown(item, x, y):
                #             transform = Transform(item.position, attitude)
                #             return (env_idx, obj_idx, transform)
        
        # No valid placement found
        return (env_idx, None, None)

    def get_action(
        self, env, obj_indices: list[list], dest_tote_ids: torch.Tensor, env_indices: torch.Tensor, plot: bool = False
    ) -> Tuple[list, torch.Tensor]:
        """Get packing actions using heuristic strategy with semantic masks."""
        
        transforms_list = []
        obj_idx_list = []
        
        # Prepare arguments for each environment
        worker_args = []
        for i, env_idx in enumerate(env_indices):
            env_idx_val = env_idx.item()
            curr_obj_indices = obj_indices[env_idx_val]
            
            if not curr_obj_indices:
                continue
            
            # Classify objects
            obj_types = [self.classify_object(env_idx_val, obj_idx) for obj_idx in curr_obj_indices]
            
            # Get target zones
            target_zones = [self.get_target_zone(obj_type) for obj_type in obj_types]
            
            # Find stacking candidates
            container = self.problems[env_idx_val].container
            stacking_candidates = [
                self.find_stacking_candidates(container, env_idx_val, obj_idx, obj_type)
                for obj_idx, obj_type in zip(curr_obj_indices, obj_types)
            ]
            
            # Get bboxes (orientations will be generated on-demand in worker)
            bbox_list = [
                self.tote_manager.get_object_bbox(env_idx_val, obj_idx).cpu().numpy()
                for obj_idx in curr_obj_indices
            ]
            
            worker_args.append({
                "env_idx": env_idx_val,
                "obj_indices": curr_obj_indices,
                "problem": self.problems[env_idx_val],
                "obj_types": obj_types,
                "target_zones": target_zones,
                "stacking_candidates": stacking_candidates,
                "bbox_list": bbox_list,
            })
        
        # Execute in parallel
        with ProcessPoolExecutor(max_workers=self.MAX_WORKERS) as executor:
            results = list(executor.map(self._heuristic_worker, worker_args))
        
        # Collect results
        for env_idx, obj_idx, transform in results:
            if obj_idx is not None and transform is not None:
                transforms_list.append(transform)
                obj_idx_list.append(torch.tensor([obj_idx], dtype=torch.int32))
                
                # Update container with semantic mask
                item = self.problems[env_idx].items[obj_idx]
                item.transform(transform)
                print(f"Adding item {obj_idx} to container {env_idx} at position {item.position} with attitude {item.attitude}")
                self.problems[env_idx].container.add_item(item, update_semantic_mask=True)
                self.packed_obj_idx[env_idx].append(torch.tensor(obj_idx))
            else:
                # Mark as unpackable
                if obj_indices[env_idx]:
                    first_obj = obj_indices[env_idx][0]
                    self.unpackable_obj_idx[env_idx].append(first_obj)
        
        if not transforms_list:
            raise ValueError("No valid transforms found for any environment.")
        
        return transforms_list, torch.cat(obj_idx_list, dim=0) if obj_idx_list else None
