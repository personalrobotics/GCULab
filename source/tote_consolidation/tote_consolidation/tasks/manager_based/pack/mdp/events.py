# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING

import isaaclab.sim as sim_utils
import torch
from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
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

    # Cache for storing volumes of already computed objects
    obj_volumes = torch.zeros((env.num_envs, num_objects), device=env.device)
    obj_bboxes = torch.zeros((env.num_envs, num_objects, 3), device=env.device)
    volume_cache = {}
    bbox_cache = {}

    # Get mesh from the asset
    for asset_cfg in asset_cfgs:
        object: RigidObject = env.scene[asset_cfg.name]
        prim_path_expr = object.cfg.prim_path  # fix prim path is regex
        for prim_path in sim_utils.find_matching_prim_paths(prim_path_expr):
            prim = env.scene.stage.GetPrimAtPath(prim_path)
            for mesh in find_meshes(prim):
                mesh_name = mesh.GetPath().__str__().split("/")[-1]  # Extract the last portion of the path
                env_idx = int(mesh.GetPath().__str__().split("/")[3].split("_")[-1])
                obj_idx = int("".join(filter(str.isdigit, mesh.GetPath().__str__().split("/")[4]))) - 1
                if mesh_name in volume_cache and mesh_name in bbox_cache:
                    volume = volume_cache[mesh_name]
                    bbox = bbox_cache[mesh_name]
                else:
                    volume = compute_mesh_volume(mesh)
                    bbox = compute_mesh_bbox(mesh)

                    # Cache the computed values
                    volume_cache[mesh_name] = volume
                    bbox_cache[mesh_name] = bbox

                obj_volumes[env_idx, obj_idx] = volume
                obj_bboxes[env_idx, obj_idx] = bbox

    env.gcu.set_object_volume(obj_volumes)
    env.gcu.set_object_bbox(obj_bboxes)
