# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math
import random
from typing import TYPE_CHECKING

import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
import torch
from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sim import schemas
from isaaclab.sim.schemas import schemas_cfg
from pxr import UsdGeom

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLGCUEnv


def object_props(
    env: ManagerBasedRLGCUEnv,
    env_ids: torch.Tensor,
    asset_cfgs: list[SceneEntityCfg] = [SceneEntityCfg("object1")],
    num_objects: int = 1,
) -> torch.Tensor:
    """The volume of the object."""

    def find_meshes(prim):
        meshes = []
        if prim.IsA(UsdGeom.Mesh):
            meshes.append(UsdGeom.Mesh(prim))
        for child in prim.GetChildren():
            meshes.extend(find_meshes(child))
        return meshes

    def compute_mesh_volume(mesh):
        """Compute volume of a triangulated, watertight mesh using PyTorch.
        Assumes 1 unit = 1 cm (0.01 meters).
        """
        meters_per_unit = 0.01  # 1 unit = 1 cm (0.01 meters)

        # Get mesh attributes
        points_attr = mesh.GetPointsAttr()
        face_counts_attr = mesh.GetFaceVertexCountsAttr()
        face_indices_attr = mesh.GetFaceVertexIndicesAttr()

        points = torch.tensor(points_attr.Get(), dtype=torch.float32)
        face_counts = face_counts_attr.Get()
        face_indices = face_indices_attr.Get()

        # Ensure all faces are triangles
        if not all(count == 3 for count in face_counts):
            raise ValueError("Mesh must be triangulated for volume calculation.")

        face_indices = torch.tensor(face_indices, dtype=torch.long).reshape(-1, 3)

        v0 = points[face_indices[:, 0]]
        v1 = points[face_indices[:, 1]]
        v2 = points[face_indices[:, 2]]

        # Use the scalar triple product: volume of tetrahedron
        cross = torch.cross(v0, v1, dim=1)
        dot = torch.sum(cross * v2, dim=1)
        volume = torch.sum(dot) / 6.0

        return volume.abs().item() * (meters_per_unit**3) * 1e6  # Convert to cubic centimeters (1 m^3 = 1e6 cm^3)

    def compute_mesh_bbox(mesh):
        """Compute the l, w, h bounding box of a mesh."""
        points_attr = mesh.GetPointsAttr()
        points = torch.tensor(points_attr.Get(), dtype=torch.float32)
        min_coords = torch.min(points, dim=0).values
        max_coords = torch.max(points, dim=0).values
        bbox = max_coords - min_coords
        bbox = bbox[[2, 0, 1]]  # Reorder to (l, w, h)
        return bbox

    def compute_voxelized_geometry(mesh, bbox, voxel_size=1, padding_factor=1.2):
        """
        Voxelize the mesh geometry into a grid where each voxel corresponds to voxel_size³ volume.
        The grid is made larger than the actual mesh by scaling up the mesh and bounding box.

        Parameters:
            mesh          : UsdGeom.Mesh
            bbox          : torch.Tensor (3,), represents the bounding box **size** of the object (max - min)
            voxel_size    : int or float, size of each voxel
            padding_factor: float, factor to scale up the bounding box (default: 1.2 = 20% larger)

        Returns:
            voxel_grid : torch.FloatTensor, shape determined by bbox and voxel_size
        """
        points_attr = mesh.GetPointsAttr()
        points = torch.tensor(points_attr.Get(), dtype=torch.float32)

        # Calculate bounding box minimum and maximum
        bbox_min = points.min(dim=0).values
        bbox_max = points.max(dim=0).values

        # Calculate center of the bounding box
        center = (bbox_min + bbox_max) / 2

        # Scale up the bounding box around its center
        scaled_min = center - (center - bbox_min) * padding_factor
        scaled_max = center + (bbox_max - center) * padding_factor

        # New scaled bbox size
        scaled_bbox = scaled_max - scaled_min

        # Compute grid size with explicit ceil
        grid_size = torch.ceil(scaled_bbox / voxel_size).long()

        voxel_grid = torch.zeros(grid_size.tolist(), dtype=torch.float32)

        # Offset points so that scaled_min maps to zero index in grid
        points_local = points - scaled_min

        for point in points_local:
            # Ensure indices are rounded properly (floor for point-to-voxel mapping)
            voxel_index = torch.floor(point / voxel_size).long()
            # Set voxel to 1 if within bounds
            if (0 <= voxel_index).all() and (voxel_index < grid_size).all():
                voxel_grid[tuple(voxel_index.tolist())] = 1.0

        # Rearrange to (z, x, y) for consistency with other functions
        voxel_grid = voxel_grid.permute(2, 0, 1)

        # Optionally visualize the voxel grid
        # voxels_np = voxel_grid.cpu().numpy()
        # fig = plt.figure(figsize=(8, 6))
        # ax = fig.add_subplot(projection='3d')
        # ax.set_box_aspect(voxels_np.shape)

        # # Plot the voxels
        # ax.voxels(voxels_np, edgecolor='k')

        # ax.set_xlabel('X')
        # ax.set_ylabel('Y')
        # ax.set_zlabel('Z')
        # plt.title('3D Voxel Grid Visualization')
        # plt.show()

        return voxel_grid

    # Cache for storing volumes of already computed objects
    obj_volumes = torch.zeros((env.num_envs, num_objects), device=env.device)
    obj_bboxes = torch.zeros((env.num_envs, num_objects, 3), device=env.device)
    obj_voxels = [[None for _ in range(num_objects)] for _ in range(env.num_envs)]
    
    mesh_properties_cache = {}

    # Get mesh from the asset
    for asset_cfg in asset_cfgs:
        object: RigidObject = env.scene[asset_cfg.name]
        prim_path_expr = object.cfg.prim_path  # fix prim path is regex
        for prim_path in sim_utils.find_matching_prim_paths(prim_path_expr):
            prim = env.scene.stage.GetPrimAtPath(prim_path)
            items = prim.GetMetadata('references').GetAddedOrExplicitItems()
            asset_path = items[0].assetPath

                
            for mesh in find_meshes(prim):
                env_idx = int(mesh.GetPath().__str__().split("/")[3].split("_")[-1])
                obj_idx = int("".join(filter(str.isdigit, mesh.GetPath().__str__().split("/")[4])))

                # Check if we've already calculated properties for this asset
                if asset_path in mesh_properties_cache:
                    volume, bbox, vox = mesh_properties_cache[asset_path]
                else:
                    volume = compute_mesh_volume(mesh)
                    bbox = compute_mesh_bbox(mesh)
                    vox = compute_voxelized_geometry(mesh, bbox)
                    mesh_properties_cache[asset_path] = (volume, bbox, vox)

                obj_volumes[env_idx, obj_idx] = volume
                obj_bboxes[env_idx, obj_idx] = bbox
                obj_voxels[env_idx][obj_idx] = vox

    env.tote_manager.set_object_volume(obj_volumes, torch.arange(env.num_envs, device=env.device))
    env.tote_manager.set_object_bbox(obj_bboxes, torch.arange(env.num_envs, device=env.device))
    env.tote_manager.set_object_voxels(obj_voxels)


def randomize_object_pose_with_invalid_ranges(
    env: ManagerBasedRLGCUEnv,
    env_ids: torch.Tensor,
    asset_cfgs: list[SceneEntityCfg],
    min_separation: float = 0.0,
    pose_range: dict[str, tuple[float, float]] = {},
    invalid_ranges: list[dict[str, tuple[float, float]]] = [],
    max_sample_tries: int = 5000,
):
    """
    Randomizes object poses while avoiding invalid pose ranges.

    Args:
        env (ManagerBasedRLGCUEnv): The environment object.
        env_ids (torch.Tensor): Tensor of environment IDs.
        asset_cfgs (list[SceneEntityCfg]): List of asset configurations.
        min_separation (float): Minimum separation between objects.
        pose_range (dict[str, tuple[float, float]]): Valid pose ranges for sampling.
        invalid_ranges (list[dict[str, tuple[float, float]]]): List of invalid pose ranges to avoid.
        max_sample_tries (int): Maximum number of sampling attempts per object.
    """
    if env_ids is None:
        return

    def sample_outside_range(valid_min, valid_max, invalid_ranges):
        """Samples a value outside of invalid ranges within the valid range."""
        # Sort invalid ranges
        if not invalid_ranges:
            return random.uniform(valid_min, valid_max)

        sorted_ranges = sorted(invalid_ranges, key=lambda x: x[0])

        # Create list of valid ranges
        valid_ranges = []
        current_min = valid_min

        for low, high in sorted_ranges:
            if current_min < low:
                valid_ranges.append((current_min, low))
            current_min = max(current_min, high)

        if current_min < valid_max:
            valid_ranges.append((current_min, valid_max))

        # No valid ranges left
        if not valid_ranges:
            return random.uniform(valid_min, valid_max)  # Fallback to original range

        # Weighted selection of range based on range size
        weights = [high - low for low, high in valid_ranges]
        if sum(weights) <= 0:
            return random.uniform(valid_min, valid_max)  # Fallback if weights are invalid

        selected_range_idx = random.choices(range(len(valid_ranges)), weights=weights, k=1)[0]
        selected_range = valid_ranges[selected_range_idx]

        return random.uniform(selected_range[0], selected_range[1])

    def get_invalid_ranges_for_dimension(dim, invalid_ranges):
        """Extract invalid ranges for a specific dimension from all invalid ranges."""
        ranges = []
        for invalid_range in invalid_ranges:
            if dim in invalid_range:
                ranges.append(invalid_range[dim])
        return ranges

    env.tote_manager.reset()

    # Randomize poses in each environment independently
    for cur_env in env_ids.tolist():
        pose_list = []
        for i in range(len(asset_cfgs)):
            for j in range(max_sample_tries):
                # Sample each dimension outside of invalid ranges
                sample = {
                    "x": sample_outside_range(
                        *pose_range.get("x", (0.0, 0.0)), get_invalid_ranges_for_dimension("x", invalid_ranges)
                    ),
                    "y": sample_outside_range(
                        *pose_range.get("y", (0.0, 0.0)), get_invalid_ranges_for_dimension("y", invalid_ranges)
                    ),
                    "z": sample_outside_range(
                        *pose_range.get("z", (0.0, 0.0)), get_invalid_ranges_for_dimension("z", invalid_ranges)
                    ),
                    "roll": sample_outside_range(
                        *pose_range.get("roll", (0.0, 0.0)), get_invalid_ranges_for_dimension("roll", invalid_ranges)
                    ),
                    "pitch": sample_outside_range(
                        *pose_range.get("pitch", (0.0, 0.0)), get_invalid_ranges_for_dimension("pitch", invalid_ranges)
                    ),
                    "yaw": sample_outside_range(
                        *pose_range.get("yaw", (0.0, 0.0)), get_invalid_ranges_for_dimension("yaw", invalid_ranges)
                    ),
                }

                # Check if pose is sufficiently separated from already sampled poses
                if len(pose_list) == 0 or all(
                    math.dist([sample["x"], sample["y"], sample["z"]], [pose["x"], pose["y"], pose["z"]])
                    > min_separation
                    for pose in pose_list
                ):
                    pose_list.append(sample)
                    break

        # Randomize pose for each object
        for i in range(len(asset_cfgs)):
            asset_cfg = asset_cfgs[i]
            asset = env.scene[asset_cfg.name]

            # Prepare poses
            pose_tensor = torch.tensor(
                [[
                    pose_list[i]["x"],
                    pose_list[i]["y"],
                    pose_list[i]["z"],
                    pose_list[i]["roll"],
                    pose_list[i]["pitch"],
                    pose_list[i]["yaw"],
                ]],
                device=env.device,
            )
            positions = pose_tensor[:, 0:3] + env.scene.env_origins[cur_env, 0:3]
            orientations = math_utils.quat_from_euler_xyz(pose_tensor[:, 3], pose_tensor[:, 4], pose_tensor[:, 5])

            env.tote_manager.update_object_positions_in_sim(
                env,
                objects=[asset_cfg.name],
                positions=positions,
                orientations=orientations,
                cur_env=cur_env,
            )
        env.tote_manager.refill_source_totes(env, env_ids=torch.arange(env.num_envs, device=env.device))


def set_objects_to_invisible(
    env: ManagerBasedRLGCUEnv,
    env_ids: torch.Tensor,
):
    """Sets the visibility of objects to false in the simulation.

    Args:
        env (ManagerBasedRLGCUEnv): The environment object.
        env_ids (torch.Tensor): Tensor of environment IDs.
        asset_cfgs (list[SceneEntityCfg]): List of asset configurations to set invisible.
    """
    if env_ids is None:
        return

    visibility_mask = ~torch.any(env.tote_manager.tote_to_obj, dim=1)

    for obj_id in range(env.tote_manager.num_objects):
        envs_obj_in_reserve = torch.nonzero(visibility_mask[:, obj_id], as_tuple=True)[0].tolist()
        if envs_obj_in_reserve:
            asset_cfg = SceneEntityCfg(f"object{obj_id}")
            asset = env.scene[asset_cfg.name]
            asset.set_visibility(False, env_ids=envs_obj_in_reserve)


def check_obj_out_of_bounds(
    env: ManagerBasedRLGCUEnv,
    env_ids: torch.Tensor,
    asset_cfgs: list[SceneEntityCfg],
):
    asset_cfgs = asset_cfgs or []
    """Check if any object is out of bounds and reset the environment if so.
    Args:
        env (ManagerBasedRLGCUEnv): The environment object.
        env_ids (torch.Tensor): Tensor of environment IDs.
        asset_cfgs (list[SceneEntityCfg]): List of asset configurations to check.
    Returns:
        bool: True if any object is out of bounds, False otherwise.
    """
    if env_ids is None:
        return False
    # Check if any object is out of bounds using asset_cfgs
    bounds = [(0.2, 0.65), (-0.8, 0.8), (0.0, 0.3)]
    for asset_cfg in asset_cfgs:
        asset = env.scene[asset_cfg.name]  # Access asset using asset_cfg
        asset_pose = asset.data.root_state_w[:, :3] - env.scene.env_origins[:, :3]
        if not all(
            bounds[i][0] <= asset_pose[:, i].min() <= bounds[i][1]
            and bounds[i][0] <= asset_pose[:, i].max() <= bounds[i][1]
            for i in range(3)
        ):
            for i in range(3):
                if not (
                    bounds[i][0] <= asset_pose[:, i].min() <= bounds[i][1]
                    and bounds[i][0] <= asset_pose[:, i].max() <= bounds[i][1]
                ):
                    print(
                        f"Asset '{asset_cfg.name}' is out of bounds on axis {i}. "
                        f"Expected bounds: {bounds[i]}, "
                        f"Found min: {asset_pose[:, i].min()}, max: {asset_pose[:, i].max()}"
                    )
            env._reset_idx(env_ids)  # Reset the environment if any object is out of bounds


def detect_objects_in_tote(env: ManagerBasedRLGCUEnv, env_ids: torch.Tensor, asset_cfgs: list[SceneEntityCfg] = []):
    """Detects objects in the tote.
    Args:
        env (ManagerBasedRLGCUEnv): The environment object.
        env_ids (torch.Tensor): Tensor of environment IDs.
        asset_cfgs (list[SceneEntityCfg]): List of asset configurations.
    """
    if env_ids is None:
        return

    for asset_cfg in asset_cfgs:
        asset = env.scene[asset_cfg.name]
        asset_pose = asset.data.root_state_w[:, :3] - env.scene.env_origins[:, :3]
        envs_in_tote = []

        for i, tote_bound in enumerate(env.tote_manager.tote_bounds):
            in_tote_envs = (
                (tote_bound[0][0] <= asset_pose[:, 0])
                & (asset_pose[:, 0] <= tote_bound[0][1])
                & (tote_bound[1][0] <= asset_pose[:, 1])
                & (asset_pose[:, 1] <= tote_bound[1][1])
                & (tote_bound[2][0] <= asset_pose[:, 2])
                & (asset_pose[:, 2] <= tote_bound[2][1])
            ).nonzero(as_tuple=True)[0]

            if len(in_tote_envs) > 0:
                envs_in_tote.extend(in_tote_envs.tolist())
                env.tote_manager.put_objects_in_tote(
                    torch.tensor([int(asset_cfg.name.split("object")[-1])], device=env.device),
                    torch.tensor([i], device=env.device),
                    in_tote_envs,
                )

        if len(envs_in_tote) == 0:
            raise RuntimeError(
                f"Object {asset_cfg.name} is not within the bounds of any tote in environments {env_ids.tolist()}. "
                f"Object pose: {asset_pose[:, :3]}, Tote bounds: {env.tote_manager.tote_bounds}"
            )
