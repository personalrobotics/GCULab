import torch

def snap_to_theta_continuous(quat: torch.Tensor, q_prev: torch.Tensor | None = None, theta: float = 22.5) -> torch.Tensor:
    """Snap a quaternion to nearest multiple of theta degrees while preserving sign continuity."""
    quat = quat / quat.norm()

    # If a previous quaternion is provided, keep consistent sign
    if q_prev is not None and torch.dot(quat, q_prev) < 0:
        quat = -quat

    # Convert to rotation matrix
    w, x, y, z = quat
    R = torch.tensor([
        [1 - 2*(y*y + z*z), 2*(x*y - z*w),     2*(x*z + y*w)],
        [2*(x*y + z*w),     1 - 2*(x*x + z*z), 2*(y*z - x*w)],
        [2*(x*z - y*w),     2*(y*z + x*w),     1 - 2*(x*x + y*y)]
    ], device=quat.device, dtype=quat.dtype)

    # Convert to Euler
    yaw = torch.atan2(R[1,0], R[0,0])
    pitch = torch.asin(-R[2,0].clamp(-1,1))
    roll = torch.atan2(R[2,1], R[2,2])

    step = theta * torch.pi / 180.0
    roll = torch.round(roll / step) * step
    pitch = torch.round(pitch / step) * step
    yaw = torch.round(yaw / step) * step

    # Convert back to quaternion
    cr, sr = torch.cos(roll/2), torch.sin(roll/2)
    cp, sp = torch.cos(pitch/2), torch.sin(pitch/2)
    cy, sy = torch.cos(yaw/2), torch.sin(yaw/2)

    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    q_snapped = torch.stack([w, x, y, z])

    # Again ensure continuity (not just canonical form)
    if q_prev is not None and torch.dot(q_snapped, q_prev) < 0:
        q_snapped = -q_snapped

    return q_snapped / q_snapped.norm()

def get_z_position_from_depth(
    image_obs: torch.Tensor, xy_pos: torch.Tensor, rotated_dim: torch.Tensor, img_h: int, img_w: int, true_tote_dim: list[float]
) -> torch.Tensor:
    """Get the z position from the depth image."""
    depth_img = image_obs.reshape(-1, img_h, img_w)

    # Rescale x_pos and y_pos to the range of the depth image
    total_tote_x = true_tote_dim[0] / 100
    total_tote_y = true_tote_dim[1] / 100
    tote_x_m = true_tote_dim[0]
    tote_y_m = true_tote_dim[1]
    x_pos = torch.round(tote_x_m - (xy_pos[0] / total_tote_x) * tote_x_m).to(torch.int64)
    y_pos = torch.round((xy_pos[1] / total_tote_y) * tote_y_m).to(torch.int64)

    # Compute patch extents in pixel units by scaling world dimensions to pixel coordinates
    # The image covers the total tote dimensions, so scale object dimensions relative to total tote dimensions
    x_extent = torch.round((rotated_dim[:, 0] / total_tote_x) * tote_x_m).clamp(min=1).long()
    y_extent = torch.round((rotated_dim[:, 1] / total_tote_y) * tote_y_m).clamp(min=1).long()

    # Compute patch start/end indices, clamp to image bounds
    x1 = x_pos.clamp(0, tote_x_m)
    y0 = y_pos.clamp(0, tote_y_m)
    x0 = (x1 - x_extent).clamp(0, tote_x_m)
    y1 = (y0 + y_extent).clamp(0, tote_y_m)

    # For each sample, extract the patch and get the max value
    # Use broadcasting to build masks for all pixels in one go
    grid_y = torch.arange(img_h, device=rotated_dim.device).view(1, img_h, 1)
    grid_x = torch.arange(img_w, device=rotated_dim.device).view(1, 1, img_w)
    y0_ = y0.view(-1, 1, 1)
    y1_ = y1.view(-1, 1, 1)
    x0_ = x0.view(-1, 1, 1)
    x1_ = x1.view(-1, 1, 1)
    mask = (grid_y >= y0_) & (grid_y <= y1_) & (grid_x >= x0_) & (grid_x <= x1_)

    # Masked min: set out-of-patch values and zeros to inf, then take min
    depth_img_masked = depth_img.clone()
    depth_img_masked[~mask] = float("inf")
    depth_img_masked[depth_img_masked == 0] = float("inf")
    z_pos = depth_img_masked.view(depth_img.shape[0], -1).min(dim=1).values

    z_pos = 20.0 - z_pos
    z_pos = z_pos.clamp(min=0.0, max=0.4)

    return z_pos
