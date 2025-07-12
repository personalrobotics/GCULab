# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import itertools
import torch
import isaaclab.utils.math as math_utils
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

def step_in_sim(env, num_steps=1):
    """
    Step the simulation for a specified number of steps.

    Args:
        env: Simulation environment object.
        num_steps (int): Number of steps to perform in the simulation.
    """
    step = env._sim_step_counter
    for _ in range(num_steps):
        env.sim.step(render=False)
        if step % env.cfg.sim.render_interval == 0:
            env.sim.render()
        step += 1
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

def update_object_positions_in_sim(env, objects, positions, orientations, cur_env):
    """
    Update object positions and orientations in the simulation.

    Args:
        env: Simulation environment object.
        objects (torch.Tensor or list): Object IDs or names to update.
        positions (torch.Tensor): Positions for each object [N, 3].
        orientations (torch.Tensor): Orientations for each object [N, 4].
        cur_env (int, torch.Tensor): Current environment ID(s).

    Returns:
        None
    """
    device = positions.device

    is_multi_env = hasattr(cur_env, "numel") and cur_env.numel() > 1

    positions = positions.to(env.device)
    orientations = orientations.to(env.device)

    # Update the object positions in the simulation
    for j, obj_id in enumerate(objects):
        # Get environment ID for this object
        if is_multi_env:
            env_id = cur_env[j].item()
        else:
            env_id = cur_env.item() if hasattr(cur_env, "item") else cur_env

        env_id_tensor = torch.tensor([env_id], device=device)

        # Get the asset based on object identifier type
        if isinstance(obj_id, str):
            asset = env.scene[obj_id]
        else:
            asset = env.scene[f"object{obj_id.item()}"]

        # Update prim path for the specific environment
        prim_path = asset.cfg.prim_path.replace("env_.*", f"env_{env_id}")

        # Modify physics properties and apply pose and velocity
        schemas.modify_rigid_body_properties(
            prim_path,
            schemas_cfg.RigidBodyPropertiesCfg(
                kinematic_enabled=False,
                disable_gravity=False,
            ),
        )

        # Apply position and orientation
        asset.write_root_link_pose_to_sim(
            torch.cat([positions[j], orientations[j]]),
            env_ids=env_id_tensor,
        )

        # Reset velocity
        asset.write_root_com_velocity_to_sim(
            torch.zeros(6, device=device),
            env_ids=env_id_tensor,
        )
        asset.set_visibility(True, env_ids=env_id_tensor)

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

def generate_positions(objects, tote_bounds, env_origin, obj_bboxes, orientations, device=None):
    """
    Generate random positions within tote bounds for objects.

    Args:
        objects (torch.Tensor): Tensor containing object IDs.
        tote_bounds (list): Bounding box limits for the tote.
        env_origin (torch.Tensor): Origin of the environment.
        obj_bboxes (torch.Tensor): Bounding boxes of objects.
        orientations (torch.Tensor): Orientations for each object.
        device (torch.device, optional): Device to use for tensors.

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

    # Adjust positions based on rotated dimensions to ensure objects fit within bounds
    x_pos = torch.rand(objects.numel(), device=device) * (x_max - x_min - rotated_dims[:, 0]) + x_min + rotated_dims[:, 0]/2
    y_pos = torch.rand(objects.numel(), device=device) * (y_max - y_min - rotated_dims[:, 1]) + y_min + rotated_dims[:, 1]/2
    z_pos = torch.rand(objects.numel(), device=device) * (z_max - z_min - rotated_dims[:, 2]) + z_min + rotated_dims[:, 2]/2

    positions = torch.stack([x_pos, y_pos, z_pos], dim=1)
    return positions + env_origin