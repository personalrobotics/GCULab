# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import itertools
import multiprocessing as mp

import isaaclab.utils.math as math_utils
import numpy as np
import torch
from isaaclab.sim import schemas
from isaaclab.sim.schemas import schemas_cfg


def calculate_rotated_bounding_box(object_bboxes, orientations, device):
    """
    Calculate rotated bounding boxes for objects efficiently.

    Args:
        object_bboxes (torch.Tensor): Bounding boxes of objects [N, 3].
        orientations (torch.Tensor): Orientation quaternions [N, 4].
        device (torch.device): Device to use for tensors.

    Returns:
        torch.Tensor: Dimensions of rotated bounding boxes.
    """
    object_half_dims = object_bboxes[:, [1, 2, 0]] / 2.0 * 0.01  # cm to m

    corners = torch.tensor(list(itertools.product([-1, 1], repeat=3)), device=device).unsqueeze(
        0
    ) * object_half_dims.unsqueeze(1)
    rot_matrices = math_utils.matrix_from_quat(orientations)
    rotated_corners = torch.bmm(corners, rot_matrices.transpose(1, 2))

    min_vals, _ = torch.min(rotated_corners, dim=1)
    max_vals, _ = torch.max(rotated_corners, dim=1)
    rotated_dims = max_vals - min_vals

    return rotated_dims


def matrix_from_quat(quaternions):
    """Convert rotations given as quaternions to rotation matrices.

    Args:
        quaternions: The quaternion orientation in (w, x, y, z). Shape is (..., 4).

    Returns:
        Rotation matrices. The shape is (..., 3, 3).

    Reference:
        https://github.com/facebookresearch/pytorch3d/blob/main/pytorch3d/transforms/rotation_conversions.py
    """
    r, i, j, k = torch.unbind(quaternions, -1)
    # Calculate two_s = 2.0 / (quaternions * quaternions).sum(-1)
    two_s = 2.0 / (quaternions * quaternions).sum(-1)

    o = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return o.reshape(quaternions.shape[:-1] + (3, 3))


def calculate_rotated_bounding_box_np(object_bboxes, orientations, device):
    """
    Calculate rotated bounding boxes for objects efficiently.

    Args:
        object_bboxes (torch.Tensor): Bounding boxes of objects [N, 3].
        orientations (torch.Tensor): Orientation quaternions [N, 4].
        device (torch.device): Device to use for tensors.

    Returns:
        torch.Tensor: Dimensions of rotated bounding boxes.
    """
    object_half_dims = object_bboxes[:, [1, 2, 0]] / 2.0 * 0.01  # cm to m

    corners = torch.tensor(list(itertools.product([-1, 1], repeat=3)), device=device).unsqueeze(
        0
    ) * object_half_dims.unsqueeze(1)
    rot_matrices = matrix_from_quat(orientations)
    rotated_corners = torch.bmm(corners, rot_matrices.transpose(1, 2)).detach().cpu()
    rotated_corners_np = rotated_corners.numpy()
    min_vals = np.min(rotated_corners_np, axis=1)
    max_vals = np.max(rotated_corners_np, axis=1)
    rotated_dims = max_vals - min_vals
    return rotated_dims


def step_in_sim(env, num_steps=1):
    """
    Step the simulation for a specified number of steps.

    Args:
        env: Simulation environment object.
        num_steps (int): Number of steps to perform in the simulation.
    """
    step = env._sim_step_counter
    for i in range(num_steps):
        env.sim.step(render=False)
        if step % env.cfg.sim.render_interval == 0:
            env.sim.render()
        step += 1
        # Only update scene on the last step to reduce GPU interface calls
        if i == num_steps - 1:
            env.scene.update(dt=env.physics_dt)


def reappear_tote_animation(env, env_ids, eject_envs, eject_tote_ids, tote_keys):
    """
    Animate totes to disappear and reappear.

    Args:
        env: Simulation environment object.
        env_ids: Tensor containing environment IDs.
        eject_envs: Tensor of boolean flags indicating environments with empty totes.
        eject_tote_ids: Tensor containing IDs of empty totes.
        tote_keys: List of tote keys.
    """
    step_in_sim(env, 20)
    for env_idx, tote_idx in zip(env_ids[eject_envs], eject_tote_ids[eject_envs]):
        if tote_idx != -1:
            tote_asset = env.scene[tote_keys[tote_idx.item()]]
            tote_asset.set_visibilities([False], [env_idx.item()])

    step_in_sim(env, 20)

    for env_idx, tote_idx in zip(env_ids[eject_envs], eject_tote_ids[eject_envs]):
        if tote_idx != -1:
            tote_asset = env.scene[tote_keys[tote_idx.item()]]
            tote_asset.set_visibilities([True], [env_idx.item()])


def calculate_tote_bounds(tote_assets, true_tote_dim, env):
    """
    Calculate bounding box limits for each tote.

    Args:
        tote_assets (list): List of tote assets in the scene.
        true_tote_dim (torch.Tensor): Dimensions of a tote in centimeters.
        env: Simulation environment object.

    Returns:
        list: Bounding box limits for each tote.
    """
    tote_bounds = []
    tote_bbox = true_tote_dim * 0.01  # Convert cm to m for bounding box calculations
    for tote in tote_assets:
        tote_pose = tote.get_world_poses()[0][0] - env.scene.env_origins[0, :3]
        bounds = [
            (tote_pose[0] - tote_bbox[0] / 2, tote_pose[0] + tote_bbox[0] / 2),
            (tote_pose[1] - tote_bbox[1] / 2, tote_pose[1] + tote_bbox[1] / 2),
            (tote_pose[2], tote_pose[2] + tote_bbox[2]),
        ]
        tote_bounds.append(bounds)
    return tote_bounds


def generate_orientations(objects, device=None):
    """
    Generate default orientations for objects.

    Args:
        objects (torch.Tensor): Tensor containing object IDs.
        device (torch.device, optional): Device to use for tensors.

    Returns:
        torch.Tensor: Orientations for the objects.
    """
    device = objects.device if device is None else device
    orientations_init = torch.tensor([1, 0, 0, 0], device=device)
    orientations_to_apply = torch.tensor([1, 0, 0, 0], device=device)

    # Create repeated orientations by multiplying the initial and to-apply quaternions
    repeated_orientations = torch.stack(
        [math_utils.quat_mul(orientations_init, orientations_to_apply)] * objects.numel()
    )

    return repeated_orientations


def generate_positions(
    objects, tote_bounds, env_origin, obj_bboxes, orientations, min_separation=0.0, device=None, max_attempts=100
):
    """
    Generate random positions within tote bounds for objects with minimum separation.

    Args:
        objects (torch.Tensor): Tensor containing object IDs.
        tote_bounds (list): Bounding box limits for the tote.
        env_origin (torch.Tensor): Origin of the environment.
        obj_bboxes (torch.Tensor): Bounding boxes of objects.
        orientations (torch.Tensor): Orientations for each object.
        min_separation (float): Minimum distance between objects (in meters).
        device (torch.device, optional): Device to use for tensors.
        max_attempts (int): Maximum number of attempts to place an object.

    Returns:
        torch.Tensor: Positions for the objects.
    """
    device = env_origin.device if device is None else device

    # Extract tote boundaries
    x_min, x_max = tote_bounds[0]
    y_min, y_max = tote_bounds[1]
    z_min, z_max = tote_bounds[2]

    # Calculate rotated bounding boxes
    rotated_dims = calculate_rotated_bounding_box(obj_bboxes, orientations, device)

    # Initialize positions list
    positions = []

    for i in range(objects.numel()):
        # Get margin to adjust boundaries for object size
        margin = rotated_dims[i] / 2.0

        # Adjusted boundaries considering object size
        adj_x_min = x_min + margin[0]
        adj_x_max = x_max - margin[0]
        adj_y_min = y_min + margin[1]
        adj_y_max = y_max - margin[1]
        adj_z_min = z_min + margin[2]
        adj_z_max = z_max - margin[2]

        for attempt in range(max_attempts):
            # Generate random position
            x = torch.rand(1, device=device) * (adj_x_max - adj_x_min) + adj_x_min
            y = torch.rand(1, device=device) * (adj_y_max - adj_y_min) + adj_y_min
            z = torch.rand(1, device=device) * (adj_z_max - adj_z_min) + adj_z_min

            candidate_position = torch.tensor([x.item(), y.item(), z.item()], device=device)

            # Accept position if it's the first object or if reached max attempts
            if len(positions) == 0 or attempt == max_attempts - 1:
                positions.append(candidate_position)
                break

            # Check separation from all previously placed objects
            separation_check = all(torch.norm(candidate_position - pos) >= min_separation for pos in positions)

            if separation_check:
                positions.append(candidate_position)
                break

    # Stack positions into a tensor
    positions_tensor = torch.stack(positions, dim=0)

    return positions_tensor + env_origin


def generate_orientations_batched(all_objects, device=None):
    """
    Generate default orientations for objects in a fully vectorized manner.

    Args:
        all_objects (list): List of object tensors for each environment.
        device (torch.device, optional): Device to use for tensors.

    Returns:
        list: Orientations for the objects in each environment.
    """
    # Get device from first non-empty tensor
    device = next((obj.device for obj in all_objects if obj.numel() > 0), device)
    if device is None:
        device = torch.device("cpu")

    orientations_init = torch.tensor([1, 0, 0, 0], device=device)
    orientations_to_apply = torch.tensor([1, 0, 0, 0], device=device)

    # Create the base orientation once
    base_orientation = math_utils.quat_mul(orientations_init, orientations_to_apply)

    orientations = []
    for objects in all_objects:
        if objects.numel() > 0:
            # Repeat the base orientation for all objects in this environment
            repeated_orientations = base_orientation.unsqueeze(0).repeat(objects.numel(), 1)
            orientations.append(repeated_orientations)
        else:
            orientations.append(torch.tensor([], device=device))

    return orientations


def generate_positions_batched(
    all_objects,
    all_tote_bounds,
    env_origins,
    all_obj_bboxes,
    all_orientations,
    min_separation=0.0,
    device=None,
    max_attempts=100,
):
    """
    Generate random positions within tote bounds for objects with minimum separation in a fully vectorized manner.

    Args:
        all_objects (list): List of object tensors for each environment.
        all_tote_bounds (list): List of tote bounds for each environment.
        env_origins (torch.Tensor): Origins of all environments.
        all_obj_bboxes (list): List of object bounding boxes for each environment.
        all_orientations (list): List of orientations for each environment.
        min_separation (float): Minimum distance between objects (in meters).
        device (torch.device, optional): Device to use for tensors.
        max_attempts (int): Maximum number of attempts to place an object.

    Returns:
        list: Positions for the objects in each environment.
    """
    device = env_origins[0].device if device is None else device
    all_positions = []

    for env_idx, (objects, tote_bounds, obj_bboxes, orientations) in enumerate(
        zip(all_objects, all_tote_bounds, all_obj_bboxes, all_orientations)
    ):
        if objects.numel() == 0:
            all_positions.append(torch.tensor([], device=device))
            continue

        # Extract tote boundaries
        x_min, x_max = tote_bounds[0]
        y_min, y_max = tote_bounds[1]
        z_min, z_max = tote_bounds[2]

        # Calculate rotated bounding boxes
        rotated_dims = calculate_rotated_bounding_box(obj_bboxes, orientations, device)

        # Calculate margins for all objects at once
        margins = rotated_dims / 2.0

        # Adjusted boundaries considering object size - vectorized
        adj_x_min = x_min + margins[:, 0]
        adj_x_max = x_max - margins[:, 0]
        adj_y_min = y_min + margins[:, 1]
        adj_y_max = y_max - margins[:, 1]
        adj_z_min = z_min + margins[:, 2]
        adj_z_max = z_max - margins[:, 2]

        # Generate all random positions at once
        x_ranges = adj_x_max - adj_x_min
        y_ranges = adj_y_max - adj_y_min
        z_ranges = adj_z_max - adj_z_min

        # Generate multiple attempts for all objects simultaneously
        x = torch.rand(max_attempts, objects.numel(), device=device) * x_ranges.unsqueeze(0) + adj_x_min.unsqueeze(0)
        y = torch.rand(max_attempts, objects.numel(), device=device) * y_ranges.unsqueeze(0) + adj_y_min.unsqueeze(0)
        z = torch.rand(max_attempts, objects.numel(), device=device) * z_ranges.unsqueeze(0) + adj_z_min.unsqueeze(0)

        # Stack into positions [max_attempts, num_objects, 3]
        candidate_positions = torch.stack([x, y, z], dim=-1)

        # Initialize final positions
        positions = torch.zeros(objects.numel(), 3, device=device)
        placed_mask = torch.zeros(objects.numel(), dtype=torch.bool, device=device)

        # For the first object, always accept the first attempt
        if objects.numel() > 0:
            positions[0] = candidate_positions[0, 0]
            placed_mask[0] = True

        # For remaining objects, check separation from all previously placed objects
        for obj_idx in range(1, objects.numel()):
            for attempt in range(max_attempts):
                candidate = candidate_positions[attempt, obj_idx]

                # Check separation from all previously placed objects - vectorized
                if obj_idx == 0 or attempt == max_attempts - 1:
                    positions[obj_idx] = candidate
                    placed_mask[obj_idx] = True
                    break

                # Calculate distances to all previously placed objects
                distances = torch.norm(candidate.unsqueeze(0) - positions[:obj_idx], dim=1)
                min_distance = torch.min(distances)

                if min_distance >= min_separation:
                    positions[obj_idx] = candidate
                    placed_mask[obj_idx] = True
                    break

        # Add environment origin
        env_origin = env_origins[env_idx] if hasattr(env_origins, "__getitem__") else env_origins
        all_positions.append(positions + env_origin)

    return all_positions


def update_object_positions_in_sim_batched(env, all_objects, all_positions, all_orientations, env_ids):
    """
    Update object positions and orientations in the simulation using batched PhysX writes.

    Args:
        env: Simulation environment object.
        all_objects (list): List of object tensors for each environment.
        all_positions (list): List of position tensors for each environment.
        all_orientations (list): List of orientation tensors for each environment.
        env_ids (torch.Tensor): Environment IDs.

    Returns:
        None
    """
    device = env_ids.device
    tote_manager = env.tote_manager

    # Collect all (env_id, obj_id, pose) tuples across all environments
    batch_env_ids = []
    batch_obj_ids = []
    batch_poses = []

    for env_idx, (objects, positions, orientations) in enumerate(zip(all_objects, all_positions, all_orientations)):
        if objects.numel() == 0:
            continue

        cur_env = env_ids[env_idx].item()
        positions = positions.to(env.device)
        orientations = orientations.to(env.device)

        for j, obj_id in enumerate(objects):
            if isinstance(obj_id, str):
                obj_idx = int(obj_id.replace("object", ""))
            else:
                obj_idx = obj_id.item()
            batch_env_ids.append(cur_env)
            batch_obj_ids.append(obj_idx)
            batch_poses.append(torch.cat([positions[j], orientations[j]]))

    if not batch_env_ids:
        return

    env_ids_tensor = torch.tensor(batch_env_ids, device=device, dtype=torch.long)
    obj_ids_tensor = torch.tensor(batch_obj_ids, device=device, dtype=torch.long)
    poses_tensor = torch.stack(batch_poses)  # (N, 7)

    # Batch write to simulation
    tote_manager._write_poses_to_sim(env_ids_tensor, obj_ids_tensor, poses_tensor)

    # Set visibility
    tote_manager.set_object_visibility_paired(True, batch_env_ids, batch_obj_ids)


def generate_positions_batched_multiprocess_cuda(
    all_objects,
    all_tote_bounds,
    env_origins,
    all_obj_bboxes,
    all_orientations,
    min_separation=0.0,
    device=None,
    max_attempts=10,
    num_processes=None,
):
    """
    Generate random positions within tote bounds for objects with minimum separation using CUDA multiprocessing.
    This version uses CUDA streams and proper GPU memory management for parallel processing.

    Args:
        all_objects (list): List of object tensors for each environment.
        all_tote_bounds (list): List of tote bounds for each environment.
        env_origins (torch.Tensor): Origins of all environments.
        all_obj_bboxes (list): List of object bounding boxes for each environment.
        all_orientations (list): List of orientations for each environment.
        min_separation (float): Minimum distance between objects (in meters).
        device (torch.device, optional): Device to use for tensors.
        max_attempts (int): Maximum number of attempts to place an object.
        num_processes (int, optional): Number of processes to use. If None, uses CPU count.

    Returns:
        list: Positions for the objects in each environment.
    """
    device = env_origins[0].device if device is None else device

    # Determine number of processes
    if num_processes is None:
        num_processes = min(mp.cpu_count(), len(all_objects))

    if num_processes <= 1 or len(all_objects) <= 1:
        # Fall back to single-threaded version for small workloads
        return generate_positions_batched(
            all_objects,
            all_tote_bounds,
            env_origins,
            all_obj_bboxes,
            all_orientations,
            min_separation,
            device,
            max_attempts,
        )

    # For CUDA, we'll use a different approach - process in chunks with CUDA streams
    if device.type == "cuda":
        return _generate_positions_batched_cuda_streams(
            all_objects,
            all_tote_bounds,
            env_origins,
            all_obj_bboxes,
            all_orientations,
            min_separation,
            device,
            max_attempts,
            num_processes,
        )

    # For CPU, use normal multiprocessing
    return generate_positions_batched_multiprocess(
        all_objects,
        all_tote_bounds,
        env_origins,
        all_obj_bboxes,
        all_orientations,
        min_separation,
        device,
        max_attempts,
        num_processes,
    )


def _generate_positions_batched_cuda_streams(
    all_objects,
    all_tote_bounds,
    env_origins,
    all_obj_bboxes,
    all_orientations,
    min_separation,
    device,
    max_attempts,
    num_processes,
):
    """
    Generate positions using CUDA streams for parallel processing on GPU.
    """
    # Split environments across chunks
    chunk_size = max(1, len(all_objects) // num_processes)
    all_positions = []

    # Create CUDA streams for parallel processing
    streams = [torch.cuda.Stream() for _ in range(num_processes)]

    # Process chunks in parallel using CUDA streams
    for i in range(0, len(all_objects), chunk_size):
        end_idx = min(i + chunk_size, len(all_objects))
        stream_idx = (i // chunk_size) % num_processes
        stream = streams[stream_idx]

        with torch.cuda.stream(stream):
            # Extract chunk data
            chunk_objects = all_objects[i:end_idx]
            chunk_tote_bounds = all_tote_bounds[i:end_idx]
            chunk_env_origins = env_origins[i:end_idx] if hasattr(env_origins, "__getitem__") else env_origins
            chunk_obj_bboxes = all_obj_bboxes[i:end_idx]
            chunk_orientations = all_orientations[i:end_idx]

            # Process this chunk using the vectorized function
            chunk_positions = generate_positions_batched(
                chunk_objects,
                chunk_tote_bounds,
                chunk_env_origins,
                chunk_obj_bboxes,
                chunk_orientations,
                min_separation,
                device,
                max_attempts,
            )

            all_positions.extend(chunk_positions)

    # Synchronize all streams
    torch.cuda.synchronize()

    return all_positions
