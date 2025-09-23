# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math
import random
from itertools import product
from typing import TYPE_CHECKING

import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
import numpy as np
import torch
import trimesh
from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sim import schemas
from isaaclab.sim.schemas import schemas_cfg
from packing3d import Container
from pxr import UsdGeom
from scipy.ndimage import binary_fill_holes, generate_binary_structure, label

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

    def compute_voxel_volume(vox: torch.Tensor) -> float:
        """Computes the volume of a voxelized object."""
        if vox is None:
            return 0.0
        # Count the number of filled voxels (1s)
        filled_voxels = torch.sum(vox > 0).item()
        # Volume is number of filled voxels times voxel size (assumed to be 1.0)
        return float(filled_voxels)

    def compute_mesh_bbox(mesh):
        """Compute the l, w, h bounding box of a mesh."""
        points_attr = mesh.GetPointsAttr()
        points = torch.tensor(points_attr.Get(), dtype=torch.float32)
        min_coords = torch.min(points, dim=0).values
        max_coords = torch.max(points, dim=0).values
        bbox = max_coords - min_coords
        bbox = bbox[[2, 0, 1]]  # Reorder to (l, w, h)
        return bbox

    def usdmesh_to_trimesh(mesh: UsdGeom.Mesh) -> trimesh.Trimesh:
        """Converts a UsdGeom.Mesh to a Trimesh object."""
        points_attr = mesh.GetPointsAttr()
        vertices = np.array(points_attr.Get(), dtype=np.float32)

        face_vertex_counts = mesh.GetFaceVertexCountsAttr().Get()
        face_vertex_indices = mesh.GetFaceVertexIndicesAttr().Get()
        assert all(c == 3 for c in face_vertex_counts), "Only triangle faces supported"
        faces = np.array(face_vertex_indices, dtype=np.int32).reshape(-1, 3)

        return trimesh.Trimesh(vertices=vertices, faces=faces, process=False)

    def compute_voxelized_geometry(mesh, bbox_size, voxel_size=1.0, padding_factor=1.15):
        """
        Voxelize the mesh solidly into a 3D grid that fits the specified bbox size (X, Y, Z).

        Args:
            mesh: The mesh to voxelize.
            bbox_size: The target bounding box size (X, Y, Z).
            voxel_size: The size of each voxel (default: 1.0).
            padding_factor: A multiplicative factor applied to the bounding box size to ensure the mesh fits comfortably
                within the voxel grid. The default value of 1.15 provides a 15% margin to account for numerical errors,
                mesh irregularities, and to avoid clipping during voxelization. Adjust as needed for tighter or looser fits.
        """
        tri_mesh = usdmesh_to_trimesh(mesh)
        if not tri_mesh.is_watertight:
            tri_mesh.fill_holes()

        # Apply padding to bbox size
        padded_bbox = np.array(bbox_size * 2, dtype=np.float32) * padding_factor

        # Compute scaling factor to fit mesh into padded bbox
        mesh_size = tri_mesh.extents
        scale = np.min(padded_bbox / mesh_size)

        # Scale and center mesh within bbox
        tri_mesh = tri_mesh.copy()
        tri_mesh.apply_scale(scale)

        mesh_min, mesh_max = tri_mesh.bounds
        mesh_center = (mesh_min + mesh_max) / 2.0
        bbox_center = padded_bbox / 2.0
        tri_mesh.apply_translation(bbox_center - mesh_center)

        # Final voxel grid size (number of voxels in each axis)
        grid_size = np.ceil(padded_bbox / voxel_size).astype(int)

        # Voxelize mesh and fill solid interior
        voxelized = tri_mesh.voxelized(pitch=voxel_size)
        matrix = voxelized.matrix.astype(np.uint8)

        # Fill holes (solidify)
        filled = binary_fill_holes(matrix).astype(np.float32)

        # Pad/crop to exact size (Z, Y, X) from computed grid_size
        current_shape = np.array(filled.shape)
        target_shape = grid_size[::-1]  # reverse to Z, Y, X

        pad_amount = np.maximum(target_shape - current_shape, 0)
        pad_width = [(0, pad) for pad in pad_amount]
        filled_padded = np.pad(filled, pad_width, mode="constant", constant_values=0)

        # Crop if oversized
        filled_cropped = filled_padded[: target_shape[0], : target_shape[1], : target_shape[2]]

        return torch.from_numpy(filled_cropped).permute(2, 0, 1).float()  # Convert to (Z, Y, X) format

    # Cache for storing volumes of already computed objects
    obj_volumes = torch.zeros((env.num_envs, num_objects), device=env.device)
    obj_bboxes = torch.zeros((env.num_envs, num_objects, 3), device=env.device)
    obj_voxels = [[None for _ in range(num_objects)] for _ in range(env.num_envs)]
    obj_asset_paths = [[None for _ in range(num_objects)] for _ in range(env.num_envs)]

    mesh_properties_cache = {}

    # Get mesh from the asset
    for asset_cfg in asset_cfgs:
        object: RigidObject = env.scene[asset_cfg.name]
        prim_path_expr = object.cfg.prim_path  # fix prim path is regex
        for prim_path in sim_utils.find_matching_prim_paths(prim_path_expr):
            prim = env.scene.stage.GetPrimAtPath(prim_path)
            items = prim.GetMetadata("references").GetAddedOrExplicitItems()
            asset_path = items[0].assetPath

            for mesh in find_meshes(prim):
                # if mesh name contains Collision, skip
                scale = 1

                if "Collisions" in mesh.GetPath().__str__():
                    continue
                if "Visuals" in mesh.GetPath().__str__():
                    scale = 100
                env_idx = int(mesh.GetPath().__str__().split("/")[3].split("_")[-1])
                obj_idx = int("".join(filter(str.isdigit, mesh.GetPath().__str__().split("/")[4])))

                # Check if we've already calculated properties for this asset
                if asset_path in mesh_properties_cache:
                    volume, bbox, vox = mesh_properties_cache[asset_path]
                else:
                    bbox = compute_mesh_bbox(mesh) * scale
                    vox = compute_voxelized_geometry(mesh, bbox)
                    volume = compute_voxel_volume(vox)
                    mesh_properties_cache[asset_path] = (volume, bbox, vox)
                obj_volumes[env_idx, obj_idx] = volume
                obj_bboxes[env_idx, obj_idx] = bbox
                obj_voxels[env_idx][obj_idx] = vox
                obj_asset_paths[env_idx][obj_idx] = asset_path

    env.tote_manager.set_object_asset_paths(obj_asset_paths, torch.arange(env.num_envs, device=env.device))
    env.tote_manager.set_object_volume(obj_volumes, torch.arange(env.num_envs, device=env.device))
    env.tote_manager.set_object_bbox(obj_bboxes, torch.arange(env.num_envs, device=env.device))
    env.tote_manager.set_object_voxels(obj_voxels)


def refill_source_totes(env: ManagerBasedRLGCUEnv, env_ids: torch.Tensor):
    """Refills the source totes with objects from the reserve."""
    if env_ids is None:
        return
    env.tote_manager.refill_source_totes(env_ids=torch.arange(env.num_envs, device=env.device)[env_ids])


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
        env.tote_manager.refill_source_totes(env_ids=torch.arange(env.num_envs, device=env.device))


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


def obs_dims(env: ManagerBasedRLGCUEnv):
    """Returns the dimensions of the object to pack."""
    if not hasattr(env, "bpp"):
        return torch.zeros((env.unwrapped.num_envs, 3), device=env.unwrapped.device)
    tote_ids = torch.zeros(env.unwrapped.num_envs, device=env.unwrapped.device).int()
    packable_objects = env.unwrapped.bpp.get_packable_object_indices(
        env.unwrapped.tote_manager.num_objects,
        env.unwrapped.tote_manager,
        torch.arange(env.unwrapped.num_envs, device=env.unwrapped.device),
        tote_ids,
    )[0]
    objs = torch.tensor([row[0] for row in packable_objects], device=env.unwrapped.device)
    obj_dims = env.unwrapped.tote_manager.obj_bboxes[
        torch.arange(env.unwrapped.num_envs, device=env.unwrapped.device), objs
    ]
    obj_dims = obj_dims / 100.0  # Convert from cm to m
    return obj_dims


def heightmap(env: ManagerBasedRLGCUEnv):
    """Creates a heightmap of the scene.
    Args:
        env (ManagerBasedRLGCUEnv): The environment object.
        env_ids (torch.Tensor): Tensor of environment IDs.
    """
    heightmap_stacked = torch.zeros(
        (env.num_envs, env.tote_manager.true_tote_dim[0], env.tote_manager.true_tote_dim[1], 1), device=env.device
    )
    # If env does not have bpp attribute or it's None, return zeros
    if not hasattr(env, "bpp"):
        return torch.zeros(
            (env.num_envs, env.tote_manager.true_tote_dim[0], env.tote_manager.true_tote_dim[1], 1), device=env.device
        )
    for env_id in range(env.num_envs):
        heightmap_stacked[env_id] = torch.from_numpy(env.bpp.problems[env_id].container.heightmap).unsqueeze(-1)
    return heightmap_stacked


def gcu_reward(env: ManagerBasedRLGCUEnv):
    """
    Computes the GCU-based reward for each environment.

    Args:
        env (ManagerBasedRLGCUEnv): The environment object.

    Returns:
        torch.Tensor: Reward tensor of shape [num_envs], where each entry corresponds to the GCU reward
                      for that environment. Only environments that are being reset receive a nonzero reward.
    """
    reset_envs = env.reset_buf.nonzero(as_tuple=False).squeeze(-1)
    rewards = torch.zeros(env.num_envs, device=env.device)
    if len(reset_envs) > 0:
        # Use the most recent GCU values for environments being reset.
        # pre_reset_rewards shape: [num_envs, num_totes]
        pre_reset_rewards = env.tote_manager.stats.recent_gcu_values
        # Assign the GCU value of the destination tote for each resetting environment.
        rewards[reset_envs] = pre_reset_rewards[reset_envs, env.tote_manager.dest_totes[reset_envs]]
    return rewards


def inverse_wasted_volume(env: ManagerBasedRLGCUEnv):
    """
    Computes the wasted volume in the tote, defined as 1 - (% top down volume - GCU of objects).
    1 - (% top down volume - GCU of objects).
    Args:
        env (ManagerBasedRLGCUEnv): The environment object.
    """
    total_volume = env.tote_manager.tote_volume
    # env.scene.write_data_to_sim()
    # env.sim.step(render=True)
    heightmaps = 20 - env.observation_manager.compute()["sensor"]  # subtract distance from camera to tote
    # import matplotlib.pyplot as plt
    # plt.imshow(heightmaps[0].cpu().numpy(), cmap='viridis')
    # plt.colorbar()
    # plt.savefig("heightmap.png")
    # plt.close()
    top_down_volumes = (0.26 - heightmaps) * 100 * 0.92  # 0.92
    top_down_volumes = torch.sum(top_down_volumes, dim=(1, 2))  # Sum over heightmap dimensions
    top_down_volumes = top_down_volumes / total_volume
    top_down_volumes = torch.clamp(top_down_volumes, min=0.0, max=1.0).squeeze(1)  # Ensure values are between 0 and 1
    objects_volume = (
        env.tote_manager.stats.recent_gcu_values[
            torch.arange(env.num_envs, device=env.device), env.tote_manager.dest_totes
        ]
        * 0.8
    )
    wasted_volume = torch.clamp(1.0 - top_down_volumes - objects_volume, min=0.0, max=1.0)
    inverse_wasted_volume = 1 / (1 + wasted_volume)  # Inverse to make it a reward
    return inverse_wasted_volume


def object_overfilled_tote(env: ManagerBasedRLGCUEnv):
    """Checks if any object is overfilled the tote.
    Args:
        env (ManagerBasedRLGCUEnv): The environment object.
        env_ids (torch.Tensor): Tensor of environment IDs.
    """
    envs_overfilled = env.tote_manager.eject_totes(
        env.tote_manager.dest_totes,
        torch.arange(env.num_envs, device=env.device),
        heightmaps=env.observation_manager._obs_buffer["sensor"],
    )
    if envs_overfilled.any():
        env.scene.write_data_to_sim()
        env.sim.step(render=True)
    return envs_overfilled


def object_shift(env: ManagerBasedRLGCUEnv):
    """Checks if any object has shifted from place pose.
    Args:
        env (ManagerBasedRLGCUEnv): The environment object.
    """
    prev_action = env.tote_manager.last_action_pos_quat
    obj_ids = prev_action[:, 0].long()  # [num_envs]
    place_action_pos_quat = prev_action[:, -7:]  # [num_envs, 7]

    # Get the settled pos quat of the objects for all envs in batch
    scene_state = env.scene.get_state(is_relative=False)
    # scene_state["rigid_object"] is a dict: { "object0": {...}, "object1": {...}, ... }
    # For each env, get the correct object key and extract its root_pose
    settled_pos_quat = torch.zeros(env.num_envs, 7, device=env.device)
    for i in range(env.num_envs):
        obj_key = f"object{int(obj_ids[i].item())}"
        settled_pos_quat[i] = scene_state["rigid_object"][obj_key]["root_pose"][i]

    pos_distance = torch.norm(place_action_pos_quat[:, :3] - settled_pos_quat[:, :3], dim=1)
    rot_distance = math_utils.quat_error_magnitude(place_action_pos_quat[:, 3:], settled_pos_quat[:, 3:])

    pos_distance_tanh = 1 - torch.tanh(pos_distance / 0.01)
    rot_distance_tanh = 1 - torch.tanh(rot_distance / 0.01)
    total_distance_tanh = pos_distance_tanh + rot_distance_tanh
    return total_distance_tanh
