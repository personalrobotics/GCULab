# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math
import os
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

    def compute_mesh_bbox(mesh):
        """Compute the l, w, h bounding box of a mesh."""
        points_attr = mesh.GetPointsAttr()
        points = torch.tensor(points_attr.Get(), dtype=torch.float32)
        min_coords = torch.min(points, dim=0).values
        max_coords = torch.max(points, dim=0).values
        bbox = max_coords - min_coords
        bbox = bbox[[2, 0, 1]]  # Reorder to (l, w, h)
        return bbox

    def compute_voxelized_geometry_usd(mesh, bbox_size, voxel_size=1.0, padding_factor=1.15, scale=1.0):
        """
        Simple USD mesh voxelization using trimesh.

        Args:
            mesh: UsdGeom.Mesh with GetPointsAttr and GetFaceVertexIndicesAttr methods
            bbox_size: Target bounding box (X, Y, Z)
            voxel_size: Voxel size (default: 1.0)
            padding_factor: Bbox padding factor (default: 1.15)
            scale: Vertex scale factor (default: 1.0)
        Returns:
            torch.FloatTensor in shape (Z, Y, X), with 1 = solid, 0 = empty
        """
        # Calculate fallback grid size
        grid_size = np.ceil(np.array(bbox_size) * padding_factor / voxel_size).astype(int)
        fallback_grid = torch.zeros(tuple(grid_size[::-1]), dtype=torch.float32)  # (Z, Y, X)

        try:
            # Extract USD mesh data
            if not (hasattr(mesh, "GetPointsAttr") and hasattr(mesh, "GetFaceVertexIndicesAttr")):
                return fallback_grid

            points = np.array(mesh.GetPointsAttr().Get(), dtype=np.float32) * scale
            face_counts = mesh.GetFaceVertexCountsAttr().Get()
            face_indices = mesh.GetFaceVertexIndicesAttr().Get()

            # Convert to triangular faces (fan triangulation for polygons)
            faces = []
            idx = 0
            for count in face_counts:
                if count >= 3:
                    face = face_indices[idx : idx + count]
                    # Fan triangulation: connect first vertex to all consecutive pairs
                    for i in range(1, count - 1):
                        faces.append([face[0], face[i], face[i + 1]])
                idx += count

            if not faces:
                return fallback_grid

            # Create and voxelize trimesh
            mesh_obj = trimesh.Trimesh(vertices=points, faces=np.array(faces))
            voxel_grid = mesh_obj.voxelized(pitch=voxel_size)

            # Convert to torch tensor
            result = torch.from_numpy(voxel_grid.matrix.astype(np.float32))

            # Uncomment to visualize:
            # plot_voxel_grid(result, f"Voxels - Scale: {scale}")

            return result

        except Exception as e:
            print(f"Voxelization failed: {e}")
            return fallback_grid

    def plot_voxel_grid(voxel_grid, title="Voxel Grid", save_path=None, bbox_size=None):
        """
        Plot a 3D voxel grid visualization.

        Args:
            voxel_grid: torch.Tensor or numpy array of shape (Z, Y, X) with 1 = solid, 0 = empty
            title: Title for the plot
            save_path: Optional path to save the plot instead of showing it
            bbox_size: Optional bounding box size (Z, Y, X) to set axis limits
        """
        try:
            import matplotlib.pyplot as plt
            from mpl_toolkits.mplot3d import Axes3D

            # Convert to numpy if it's a torch tensor
            if isinstance(voxel_grid, torch.Tensor):
                voxel_data = voxel_grid.cpu().numpy()
            else:
                voxel_data = voxel_grid

            # Get indices of non-zero voxels
            z, y, x = np.nonzero(voxel_data)

            if len(x) == 0:
                print("Warning: Voxel grid is empty, nothing to plot")
                return

            # Create 3D plot
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection="3d")

            # Plot voxels as scatter points
            ax.scatter(x, y, z, c="red", marker="s", s=20, alpha=0.8)

            # Set labels and title
            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            ax.set_zlabel("Z")
            ax.set_title(title)

            # Set axis limits based on bbox_size if provided, otherwise use voxel grid shape
            if bbox_size is not None:
                # bbox_size is (X, Y, Z), but voxel_data is (Z, Y, X)
                ax.set_xlim([0, bbox_size[0]])
                ax.set_ylim([0, bbox_size[1]])
                ax.set_zlim([0, bbox_size[2]])
                print("Setting axis limits based on provided bounding box size ", bbox_size)
            else:
                # Fall back to voxel grid dimensions
                max_range = max(voxel_data.shape)
                ax.set_xlim([0, max_range])
                ax.set_ylim([0, max_range])
                ax.set_zlim([0, max_range])

            # Add grid info to title
            shape_str = f"Shape: {voxel_data.shape}, Filled: {len(x)} voxels"
            ax.set_title(f"{title}\n{shape_str}")

            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches="tight")
                print(f"Voxel plot saved to: {save_path}")
            else:
                plt.show()

        except ImportError:
            print("Warning: matplotlib not available, cannot plot voxel grid")
        except Exception as e:
            print(f"Warning: Failed to plot voxel grid: {e}")

    def mesh_volume(mesh: UsdGeom.Mesh) -> float:
        points = np.array(mesh.GetPointsAttr().Get())
        faces = np.array(mesh.GetFaceVertexIndicesAttr().Get())
        counts = np.array(mesh.GetFaceVertexCountsAttr().Get())

        # triangulate if needed
        tris = []
        i = 0
        for c in counts:
            for j in range(1, c - 1):
                tris.append([faces[i], faces[i + j], faces[i + j + 1]])
            i += c
        tris = np.array(tris)

        # compute signed volume from triangles
        v0 = points[tris[:, 0]]
        v1 = points[tris[:, 1]]
        v2 = points[tris[:, 2]]
        signed_vol = np.sum(np.einsum("ij,ij->i", v0, np.cross(v1, v2))) / 6.0
        return abs(signed_vol)

    def load_latents(asset_path):
        """Loads the latents for a mesh.
        they are stored in the same directory as the mesh, in subfolder latents and _latent.pt suffix
        """
        # Get the directory of the asset path
        asset_dir = os.path.dirname(asset_path)
        # Get the filename without extension
        asset_filename = os.path.basename(asset_path).replace(".usd", "")
        # Construct path to latents subfolder
        latents_path = os.path.join(asset_dir, "latents", f"{asset_filename}_latent.pt")
        if not os.path.exists(latents_path):
            print(f"WARNING: Latents file not found at: {latents_path}")
            return None
        latents = torch.load(latents_path)
        return latents

    # Cache for storing volumes of already computed objects
    obj_asset_paths = [[None for _ in range(num_objects)] for _ in range(env.num_envs)]

    # Cache for storing unique object properties by asset path
    mesh_properties_cache = {}

    # Get mesh from the asset and populate asset paths
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

                # Store asset path for this object
                obj_asset_paths[env_idx][obj_idx] = asset_path

                # Compute properties only once per unique asset path
                if asset_path not in mesh_properties_cache:
                    print("asset_path", asset_path)
                    bbox = compute_mesh_bbox(mesh) * scale
                    print("bbox: ", bbox)
                    vox = compute_voxelized_geometry_usd(mesh, bbox, scale=scale)
                    volume = mesh_volume(mesh) * (scale**3)
                    latents = load_latents(asset_path)
                    mesh_properties_cache[asset_path] = (volume, bbox, vox, latents)
    # Store only the unique properties and asset paths in tote_manager
    env.tote_manager.set_object_asset_paths(obj_asset_paths, torch.arange(env.num_envs, device=env.device))
    env.tote_manager.set_unique_object_properties(mesh_properties_cache)


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

    # Update FIFO queues and peek at the first object in each queue
    env.unwrapped.bpp.update_fifo_queues(packable_objects)
    objs = torch.tensor(
        [
            (
                env.unwrapped.bpp.fifo_queues[env_idx][0].item()
                if env.unwrapped.bpp.fifo_queues[env_idx]
                else (packable_objects[env_idx][0].item() if packable_objects[env_idx] else 0)
            )
            for env_idx in range(env.unwrapped.num_envs)
        ],
        device=env.unwrapped.device,
    )
    obj_dims = torch.stack([
        env.unwrapped.tote_manager.get_object_bbox(env_idx, obj.item())
        for env_idx, obj in zip(torch.arange(env.unwrapped.num_envs, device=env.unwrapped.device), objs)
    ])
    obj_dims = obj_dims / 100.0  # Convert from cm to m
    return obj_dims


def obs_lookahead(env: ManagerBasedRLGCUEnv, max_objects: int = 20):
    """Returns the dimensions of all objects to pack, padded to 20 per environment, flattened to (num_envs, max_objects*3)."""
    if not hasattr(env, "bpp"):
        return torch.zeros((env.unwrapped.num_envs, max_objects * 3), device=env.unwrapped.device)
    tote_ids = torch.zeros(env.unwrapped.num_envs, device=env.unwrapped.device).int()
    packable_objects = env.unwrapped.bpp.get_packable_object_indices(
        env.unwrapped.tote_manager.num_objects,
        env.unwrapped.tote_manager,
        torch.arange(env.unwrapped.num_envs, device=env.unwrapped.device),
        tote_ids,
    )[0]

    # Update FIFO queues
    env.unwrapped.bpp.update_fifo_queues(packable_objects)

    obj_dims_list = []
    for env_idx in range(env.unwrapped.num_envs):
        obj_indices = []
        # Use FIFO queue if available, else fall back to packable_objects
        if env.unwrapped.bpp.fifo_queues[env_idx]:
            obj_indices = [obj.item() for obj in env.unwrapped.bpp.fifo_queues[env_idx]]
        elif packable_objects[env_idx]:
            obj_indices = [obj.item() for obj in packable_objects[env_idx]]
        # Pad or truncate to max_objects
        obj_indices = obj_indices[:max_objects]
        obj_indices += [0] * (max_objects - len(obj_indices))
        # Get dimensions for each object
        dims = [env.unwrapped.tote_manager.get_object_bbox(env_idx, obj_id) for obj_id in obj_indices]
        dims = torch.stack(dims) if dims else torch.zeros((max_objects, 3), device=env.unwrapped.device)
        obj_dims_list.append(dims)

    obj_dims = torch.stack(obj_dims_list)  # (num_envs, max_objects, 3)
    obj_dims = obj_dims / 100.0  # Convert from cm to m
    obj_dims = obj_dims.reshape(env.unwrapped.num_envs, max_objects * 3)
    return obj_dims


def obs_latents(env: ManagerBasedRLGCUEnv):
    """Returns the latents of the object to pack."""
    if not hasattr(env, "bpp"):
        return torch.zeros((env.unwrapped.num_envs, 512, 8), device=env.unwrapped.device)
    tote_ids = torch.zeros(env.unwrapped.num_envs, device=env.unwrapped.device).int()
    packable_objects = env.unwrapped.bpp.get_packable_object_indices(
        env.unwrapped.tote_manager.num_objects,
        env.unwrapped.tote_manager,
        torch.arange(env.unwrapped.num_envs, device=env.unwrapped.device),
        tote_ids,
    )[0]
    # Update FIFO queues and peek at the first object in each queue
    env.unwrapped.bpp.update_fifo_queues(packable_objects)
    objs = torch.tensor(
        [
            (
                env.unwrapped.bpp.fifo_queues[env_idx][0].item()
                if env.unwrapped.bpp.fifo_queues[env_idx]
                else (packable_objects[env_idx][0].item() if packable_objects[env_idx] else 0)
            )
            for env_idx in range(env.unwrapped.num_envs)
        ],
        device=env.unwrapped.device,
    )
    obj_latents = torch.stack([
        env.unwrapped.tote_manager.get_object_latents(env_idx, obj.item())
        for env_idx, obj in zip(torch.arange(env.unwrapped.num_envs, device=env.unwrapped.device), objs)
    ]).reshape(env.unwrapped.num_envs, -1)
    return obj_latents


def obj_ids(env: ManagerBasedRLGCUEnv):
    """Returns the asset IDs of the objects to pack."""
    if not hasattr(env, "bpp"):
        return torch.zeros((env.unwrapped.num_envs, 1), device=env.unwrapped.device)

    tote_ids = torch.zeros(env.unwrapped.num_envs, device=env.unwrapped.device).int()
    packable_objects = env.unwrapped.bpp.get_packable_object_indices(
        env.unwrapped.tote_manager.num_objects,
        env.unwrapped.tote_manager,
        torch.arange(env.unwrapped.num_envs, device=env.unwrapped.device),
        tote_ids,
    )[0]

    # Efficient lookup using pre-computed cache
    bpp = env.unwrapped.bpp
    env_indices = []
    obj_indices = []

    for env_idx, obj_list in enumerate(packable_objects):
        if len(obj_list) > 0:
            env_indices.append(env_idx)
            obj_indices.append(obj_list[0].item())
        else:
            env_indices.append(env_idx)
            obj_indices.append(0)  # Default object ID

    # Use efficient batch lookup
    asset_ids = bpp.get_asset_ids_batch(env_indices, obj_indices)
    return asset_ids.unsqueeze(1)


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


def gcu_reward_step(env: ManagerBasedRLGCUEnv):
    """
    Computes the stepwise GCU reward for each environment: the increase in GCU since the last step.
    Returns 0 for environments being reset.

    Args:
        env (ManagerBasedRLGCUEnv): The environment object.

    Returns:
        torch.Tensor: Reward tensor of shape [num_envs], where each entry is the GCU increase for that environment,
                        or 0 if the environment is being reset.
    """
    tote_ids = torch.zeros(env.unwrapped.num_envs, device=env.unwrapped.device).int()
    packable_objects = env.unwrapped.bpp.get_packable_object_indices(
        env.unwrapped.tote_manager.num_objects,
        env.unwrapped.tote_manager,
        torch.arange(env.unwrapped.num_envs, device=env.unwrapped.device),
        tote_ids,
    )[0]
    objs = torch.tensor([row[0] for row in packable_objects], device=env.unwrapped.device)
    obj_dims = torch.stack([
        env.unwrapped.tote_manager.get_object_bbox(env_idx, obj.item())
        for env_idx, obj in zip(torch.arange(env.unwrapped.num_envs, device=env.unwrapped.device), objs)
    ])
    obj_dims = obj_dims / 100.0  # Convert from cm to m
    tote_dims = env.tote_manager.true_tote_dim / 100.0
    next_box_vol = obj_dims[:, 0] * obj_dims[:, 1] * obj_dims[:, 2]
    tote_vol = torch.prod(tote_dims, dim=-1)
    gcu_values = next_box_vol / tote_vol

    # Set reward to 0 for environments that are being reset
    reset_envs = env.reset_buf.nonzero(as_tuple=False).squeeze(-1)
    if len(reset_envs.shape) == 0:
        reset_envs = reset_envs.unsqueeze(0)
    gcu_values[reset_envs] = 0.0
    return gcu_values


def _get_dest_tote_gcu(env: ManagerBasedRLGCUEnv, env_ids: torch.Tensor) -> torch.Tensor:
    gcu_values = env.unwrapped.tote_manager.get_gcu(env_ids)  # Shape: [len(env_ids), num_totes]
    dest_totes = env.unwrapped.tote_manager.dest_totes[env_ids]  # Shape: [len(env_ids)]
    envs = torch.arange(env_ids.numel(), device=env_ids.device)
    return gcu_values[envs, dest_totes]


def log_gcu_dest_tote(env: ManagerBasedRLGCUEnv, env_ids: torch.Tensor | None = None):
    """
    Computes the GCU of the destination tote for each environment.
    This tracks the GCU of the tote being actively packed. 
    This gets logged as the average GCU across envs
    
    Args:
        env (ManagerBasedRLGCUEnv): The environment object.
        
    Returns:
        torch.Tensor: GCU of destination tote per environment, shape [num_envs]
    """
    dest_gcu = _get_dest_tote_gcu(env, env_ids)
    log = env.extras.setdefault("log", {})
    log["GCU/gcu_dest_tote"] = dest_gcu


def log_gcu_max(env: ManagerBasedRLGCUEnv, env_ids: torch.Tensor | None = None):
    """
    Computes the maximum GCU of the destination tote across environments
    
    Args:
        env (ManagerBasedRLGCUEnv): The environment object.
        
    Returns:
        torch.Tensor: GCU of destination tote per environment, shape [num_envs]
    """
    dest_gcu = _get_dest_tote_gcu(env, env_ids)
    max_gcu = dest_gcu.max()
    log = env.extras.setdefault("log", {})
    log["GCU/gcu_max"] = max_gcu

def inverse_wasted_volume(env: ManagerBasedRLGCUEnv, gamma=0.99):
    """
    Computes the wasted volume in the tote, defined as 1 - (% top down volume - GCU of objects).
    1 - (% top down volume - GCU of objects).
    Args:
        env (ManagerBasedRLGCUEnv): The environment object.
    """
    total_volume = env.tote_manager.tote_volume
    heightmaps = 19.99 - env.observation_manager.compute()["sensor"]  # subtract distance from camera to tote
    top_down_volumes_ = torch.clamp(heightmaps * 100, min=0)  # Ensure no negative values
    top_down_volumes = torch.sum(top_down_volumes_, dim=(1, 2))  # Sum over heightmap dimensions

    top_down_volumes = (top_down_volumes / total_volume).squeeze(1)
    objects_volume = env.tote_manager.stats.recent_gcu_values[
        torch.arange(env.num_envs, device=env.device), env.tote_manager.dest_totes
    ]
    inverse_wasted_volume = objects_volume / top_down_volumes
    return inverse_wasted_volume


def wasted_volume_pbrs(env: ManagerBasedRLGCUEnv, gamma=0.99):
    """
    Computes the wasted volume in the tote, defined as 1 - (% top down volume - GCU of objects).
    1 - (% top down volume - GCU of objects).
    Args:
        env (ManagerBasedRLGCUEnv): The environment object.
    """
    last_pbrs = env.tote_manager.last_pbrs
    total_volume = env.tote_manager.tote_volume
    heightmaps = 19.99 - env.observation_manager.compute()["sensor"]  # subtract distance from camera to tote
    # import matplotlib.pyplot as plt
    # plt.imshow(heightmaps[0].cpu().numpy())
    # plt.show()
    top_down_volumes_ = torch.clamp(heightmaps * 100, min=0)  # Ensure no negative values
    # plt.imshow(top_down_volumes_[0].cpu().numpy())
    # plt.show()
    top_down_volumes = torch.sum(top_down_volumes_, dim=(1, 2))  # Sum over heightmap dimensions

    top_down_volumes = (top_down_volumes / total_volume).squeeze(1)
    objects_volume = env.tote_manager.stats.recent_gcu_values[
        torch.arange(env.num_envs, device=env.device), env.tote_manager.dest_totes
    ]
    inverse_wasted_volume = objects_volume / (top_down_volumes + 1e-6)
    pbrs = gamma * inverse_wasted_volume - last_pbrs
    env.tote_manager.last_pbrs = inverse_wasted_volume
    if env.tote_manager.reset_pbrs.any():
        env.tote_manager.last_pbrs[env.tote_manager.reset_pbrs] = 0
        env.tote_manager.reset_pbrs[env.tote_manager.reset_pbrs] = False
    return pbrs


def object_overfilled_tote(env: ManagerBasedRLGCUEnv):
    """Checks if any object is overfilled the tote.
    Args:
        env (ManagerBasedRLGCUEnv): The environment object.
        env_ids (torch.Tensor): Tensor of environment IDs.
    """
    env.observation_manager.compute()
    envs_overfilled = env.tote_manager.eject_totes(
        env.tote_manager.dest_totes,
        torch.arange(env.num_envs, device=env.device),
        heightmaps=env.observation_manager._obs_buffer["sensor"],
    )
    if envs_overfilled.any():
        env_ids = torch.arange(env.num_envs, device=env.device)
        env.scene.write_data_to_sim()
        env.sim.render()
        env.tote_manager.reset_pbrs[env_ids[envs_overfilled]] = True
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
