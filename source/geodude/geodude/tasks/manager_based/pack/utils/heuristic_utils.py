import math
import re
from typing import Any, Dict, List, Optional, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy

import numpy as np
import torch
from scipy import ndimage
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

    # Grid search parameters for last resort search
    GRID_SEARCH_NUM = 25
    STEP_WIDTH = 90
    
    # Object classification by YCB ID
    OBJECT_TYPE_MAP = {
        "003": ObjectType.CUBOIDAL,      # cracker_box
        "004": ObjectType.CUBOIDAL,      # sugar_box
        "010": ObjectType.CYLINDRICAL,   # potted_meat_can
        "011": ObjectType.BANANA,        # banana
        "024": ObjectType.BOWL,          # bowl
        "025": ObjectType.MUG,           # mug
    }
    
    # Object types that should prioritize stacking before grid search
    STACKABLE_TYPES = {
        ObjectType.CYLINDRICAL,
        ObjectType.BOWL,
        ObjectType.MUG,
    }

    def __init__(self, tote_manager, num_envs: int, objects: list, scale: float = 1.0, **kwargs):
        super().__init__(tote_manager, num_envs, objects, scale, **kwargs)
        # Track most recent placement position for stackable objects: {env_idx: {obj_type: (x, y)}}
        self.stacking_positions: Dict[int, Dict[str, Tuple[int, int]]] = {
            i: {} for i in range(num_envs)
        }
    
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

    def find_stacking_candidates(self, env_idx: int, obj_type: str) -> List[Tuple[int, int]]:
        """Find the stacking position for this object type.
        
        Returns a single-element list with (x, y) if a stacking position
        exists for this object type, otherwise empty list.
        """
        env_positions = self.stacking_positions.get(env_idx, {})
        position = env_positions.get(obj_type)
        
        if position is None:
            return []
        
        return [position]

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
                orientations.append(Attitude(roll=0, pitch=0, yaw=0))
            else:  # z is longest (pitch rotation "stands it up"). 
                # I think we only have objects where z is longest
                orientations.append(Attitude(roll=0, pitch=90, yaw=90)) # prioritize axis aligned with shorter axis
                orientations.append(Attitude(roll=0, pitch=90, yaw=0)) # axis aligned with longer axis
            
            # Add various flat orientations as backup
            orientations.append(Attitude(roll=0, pitch=0, yaw=0))
            orientations.append(Attitude(roll=90, pitch=0, yaw=0))
            orientations.append(Attitude(roll=0, pitch=90, yaw=0))
            
        elif obj_type == ObjectType.CYLINDRICAL:
            # Cylinders: try vertical (standing)
            orientations.append(Attitude(roll=0, pitch=90, yaw=0))
            
        elif obj_type == ObjectType.BOWL:
            # Bowls: face up for nesting
            orientations.append(Attitude(roll=0, pitch=90, yaw=0))
            
        elif obj_type == ObjectType.MUG:
            # Mugs: face down for stacking
            #TODO: need to flip handle around
            orientations.append(Attitude(roll=0, pitch=-90, yaw=-135))
            
        elif obj_type == ObjectType.BANANA:
            # Bananas: lay flat
            orientations.append(Attitude(roll=90, pitch=90, yaw=0))
            orientations.append(Attitude(roll=90, pitch=0, yaw=90))
            
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
    def _is_stackable(obj_type: str) -> bool:
        """Check if an object type should prioritize stacking."""
        return obj_type in Heuristic.STACKABLE_TYPES

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
            is_stackable = Heuristic._is_stackable(obj_type)
            
            # Generate orientations on-demand
            orientations = Heuristic._generate_orientations_static(bbox, obj_type)
            
            # Try each orientation
            for attitude in orientations:
                item.rotate(attitude)
                item.calc_heightmap()

                # Priority 1: Try stacking on similar objects (only for stackable types)
                if is_stackable:
                    for (sx, sy) in stacking_cands:
                        print(f"trying to stack object {obj_idx} ({obj_type}) in environment {env_idx} at position {sx}, {sy}")
                        if problem.container.add_item_topdown(item, sx, sy):
                            transform = Transform(item.position, attitude)
                            return (env_idx, obj_idx, transform)
                
                # Priority 2: Try target zone with type-specific heuristic
                zone_x_min, zone_x_max = target_zone
                best_score = float('inf')
                best_position = None
                
                # Grid search in target zone
                deepest_top_right = False
                for x in range(zone_x_min, min(zone_x_max, problem.container.geometry.x_size)):
                    for y in range(problem.container.geometry.y_size):
                        if problem.container.add_item_topdown(item, x, y):
                            if obj_type == ObjectType.CUBOIDAL:
                                # DBLF (Deep Bottom Left First): prioritize low z, then low y, then low x
                                score = item.position.z * 100  + y * 10 + x
                            else:
                                deepest_top_right = True
                                # DTR (Deep Top Right): prioritize low z, then high x, then high y
                                # Negate x and y to maximize them while minimizing overall score
                                max_x = problem.container.geometry.x_size
                                max_y = problem.container.geometry.y_size
                                score = item.position.z * 100  + (max_y - y) * 10 + (max_x - x)
                            
                            if score < best_score:
                                best_score = score
                                best_position = Transform(item.position, attitude)
                
                if best_position is not None:
                    print(f"Region search with region Deepest top right {deepest_top_right} found best position {best_position}")
                    return (env_idx, obj_idx, best_position)
                
                # Priority 3: DBLF grid search (Deep Bottom Left First)
                print(f"DBLF grid search: placing object {obj_idx} ({obj_type}) in environment {env_idx}")
                best_score = float('inf')
                best_transform = None
                
                for x in range(problem.container.geometry.x_size):
                    for y in range(problem.container.geometry.y_size):
                        if problem.container.add_item_topdown(item, x, y):
                            z = item.position.z
                            # DBLF scoring: prioritize bottom (low z), then front (low y), then back/left (low x)
                            score = z * 10000 + y * 100 + x
                            if score < best_score:
                                best_score = score
                                best_transform = Transform(Position(x, y, z), attitude)
                
                if best_transform is not None:
                    print(f"  -> Best DBLF position: ({best_transform.position.x}, {best_transform.position.y}, {best_transform.position.z}) score={best_score}")
                    return (env_idx, obj_idx, best_transform)

                # Priority 4: Last resort search
                transforms = problem.container.search_possible_position(
                    item, grid_num=Heuristic.GRID_SEARCH_NUM, step_width=Heuristic.STEP_WIDTH
                )
                if transforms:
                    print(f"Last resort search found best position {transforms[0]}")
                    return (env_idx, obj_idx, transforms[0])
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
            target_zones = [self.CUBOIDAL_ZONE if obj_type in ObjectType.CUBOIDAL else self.IRREGULAR_ZONE for obj_type in obj_types]
            
            # Find stacking candidates
            stacking_candidates = [
                self.find_stacking_candidates(env_idx_val, obj_type)
                for obj_type in obj_types
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
                
                # Update container
                item = self.problems[env_idx].items[obj_idx]
                item.transform(transform)
                print(f"Adding item {obj_idx} to container {env_idx} at position {item.position} with attitude {item.attitude}")
                self.problems[env_idx].container.add_item(item, update_semantic_mask=True)
                self.packed_obj_idx[env_idx].append(torch.tensor(obj_idx))
                np.set_printoptions(threshold=np.inf, linewidth=200)
                print(self.get_semantic_mask(env_idx).T)
                
                # Track most recent stacking position for stackable objects
                ycb_id = item.obj_id
                obj_type = self.classify_object_by_ycb_id(ycb_id)
                if obj_type in self.STACKABLE_TYPES:
                    self.stacking_positions[env_idx][obj_type] = (
                        transform.position.x, transform.position.y
                    )
            else:
                # Mark as unpackable
                if obj_indices[env_idx]:
                    first_obj = obj_indices[env_idx][0]
                    self.unpackable_obj_idx[env_idx].append(first_obj)
        
        if not transforms_list:
            raise ValueError("No valid transforms found for any environment.")
        
        return transforms_list, torch.cat(obj_idx_list, dim=0) if obj_idx_list else None
