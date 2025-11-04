# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to run an environment with zero action agent."""

"""Launch Isaac Sim Simulator first."""

import argparse
from collections.abc import Callable

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Zero agent for Isaac Lab environments.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--teleop_device",
    type=str,
    default="keyboard",
    help="Device for interacting with environment. Examples: keyboard, spacemouse, gamepad, handtracking, manusvive",
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--sensitivity", type=float, default=1.0, help="Sensitivity factor.")
parser.add_argument(
    "--dataset_file", type=str, default="./datasets/dataset.hdf5", help="File path to export recorded demos."
)

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import os

import gymnasium as gym
import isaaclab.utils.math as math_utils
import isaaclab_tasks  # noqa: F401
import numpy as np
import omni.log
import torch
import tote_consolidation.tasks  # noqa: F401
from gculab.devices import Se3Keyboard, Se3KeyboardCfg
from gculab.envs.mdp.recorders.recorders_cfg import (
    ActionStateSensorObservationsRecorderManagerCfg,
)
from isaaclab.devices import Se3SpaceMouse, Se3SpaceMouseCfg
from isaaclab.devices.teleop_device_factory import create_teleop_device
from isaaclab.managers import DatasetExportMode
from isaaclab.utils.math import quat_mul
from isaaclab_tasks.utils import parse_env_cfg
from isaacsim.core.api.objects import cuboid
from pack_helpers import get_z_position_from_depth, snap_to_theta_continuous
from tensordict import TensorDict
from tote_consolidation.tasks.manager_based.pack.utils.tote_helpers import (
    calculate_rotated_bounding_box,
)


def setup_output_directories() -> tuple[str, str]:
    """Set up output directories for saving demonstrations.

    Creates the output directory if it doesn't exist and extracts the file name
    from the dataset file path.

    Returns:
        tuple[str, str]: A tuple containing:
            - output_dir: The directory path where the dataset will be saved
            - output_file_name: The filename (without extension) for the dataset
    """
    # get directory path and file name (without extension) from cli arguments
    output_dir = os.path.dirname(args_cli.dataset_file)
    output_file_name = os.path.splitext(os.path.basename(args_cli.dataset_file))[0]

    # create directory if it does not exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")

    return output_dir, output_file_name


def main():
    """Zero actions agent with Isaac Lab environment."""
    # parse configuration
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )

    output_dir, output_file_name = setup_output_directories()

    env_cfg.recorders: ActionStateSensorObservationsRecorderManagerCfg = (
        ActionStateSensorObservationsRecorderManagerCfg()
    )
    env_cfg.recorders.dataset_export_dir_path = output_dir
    env_cfg.recorders.dataset_filename = output_file_name
    env_cfg.recorders.dataset_export_mode = DatasetExportMode.EXPORT_ALL

    env_cfg.seed = args_cli.seed

    # create environment
    env = gym.make(args_cli.task, cfg=env_cfg)

    target = cuboid.VisualCuboid(
        "/World/target",
        position=np.array([0.5, -0.7, 0.2]),
        orientation=np.array([0, 0, 0, 1]),
        color=np.array([1.0, 0, 0]),
        size=0.05,
        visible=False,
    )

    # print info (this is vectorized environment)
    print(f"[INFO]: Gym observation space: {env.observation_space}")
    print(f"[INFO]: Gym action space: {env.action_space}")

    past_pose = None
    target_pose = None

    object_to_pack = None
    last_objects_positions = {}
    last_placed_object = None
    get_new_obj = True

    image_obs = None
    img_h = env.unwrapped.observation_space["sensor"].shape[-3]
    img_w = env.unwrapped.observation_space["sensor"].shape[-2]

    # Flags for controlling teleoperation flow
    should_reset_recording_instance = False
    teleoperation_active = True

    # Callback handlers
    def reset_recording_instance() -> None:
        """
        Reset the environment to its initial state.

        Sets a flag to reset the environment on the next simulation step.

        Returns:
            None
        """
        nonlocal should_reset_recording_instance
        should_reset_recording_instance = True
        print("Reset triggered - Environment will reset on next step")

    def start_teleoperation() -> None:
        """
        Activate teleoperation control of the robot.

        Enables the application of teleoperation commands to the environment.

        Returns:
            None
        """
        nonlocal teleoperation_active
        teleoperation_active = True
        print("Teleoperation activated")

    def stop_teleoperation() -> None:
        """
        Deactivate teleoperation control of the robot.

        Disables the application of teleoperation commands to the environment.

        Returns:
            None
        """
        nonlocal teleoperation_active
        teleoperation_active = False
        print("Teleoperation deactivated")

    def place_object() -> None:
        nonlocal last_objects_positions
        nonlocal last_placed_object
        nonlocal get_new_obj
        nonlocal object_to_pack

        nonlocal image_obs
        nonlocal img_h, img_w

        for i in range(env.unwrapped.tote_manager.num_objects):
            asset = env.unwrapped.scene[f"object{i}"]
            pose = asset.data.root_pose_w.squeeze(0)
            last_objects_positions[i] = pose

        last_placed_object = object_to_pack

        env.unwrapped.bpp.remove_selected_from_fifo(object_to_pack)

        for i in range(env.unwrapped.num_envs):
            env.unwrapped.bpp.packed_obj_idx[i].append(
                torch.tensor([object_to_pack[i].item()], device=env.unwrapped.device)
            )

        tote_ids = torch.tensor([0], device=env.unwrapped.device).int()
        x_cube, y_cube, z_cube = cube_position[0].item(), cube_position[1].item(), cube_position[2].item()

        x_tote, y_tote, z_tote = env.unwrapped.tote_manager._tote_assets_state[0, tote_ids.item()][0:3]

        x_tote_dim, y_tote_dim = (
            env.unwrapped.tote_manager.true_tote_dim[0] / 100,
            env.unwrapped.tote_manager.true_tote_dim[1] / 100,
        )

        rotated_dim = calculate_rotated_bounding_box(
            env.unwrapped.tote_manager.get_object_bbox(0, object_to_pack).unsqueeze(0),
            cube_orientation.unsqueeze(0),
            device=env.unwrapped.device,
        )

        x_diff = x_cube - x_tote + x_tote_dim / 2 - rotated_dim[0, 0] / 2
        y_diff = y_cube - y_tote + y_tote_dim / 2 - rotated_dim[0, 1] / 2
        z_diff = z_cube - z_tote

        if image_obs is None:
            raise ValueError("Depth image observation is None. Cannot compute z position.")
        z_pos = (
            get_z_position_from_depth(
                image_obs, [x_diff, y_diff], rotated_dim, img_h, img_w, env.unwrapped.tote_manager.true_tote_dim
            )
            + 0.01
        )
        actions = torch.tensor(
            [x_diff, y_diff, z_pos, cube_orientation[0], cube_orientation[1], cube_orientation[2], cube_orientation[3]],
            device=env.unwrapped.device,
        ).unsqueeze(0)
        actions = torch.cat(
            [
                tote_ids.unsqueeze(1).to(env.unwrapped.device),  # Destination tote IDs
                object_to_pack.unsqueeze(1),  # Object indices
                actions,
            ],
            dim=1,
        )
        actions[:, 4] = z_pos

        env.unwrapped.step(actions)
        get_new_obj = True

    def undo_place() -> None:
        nonlocal last_objects_positions
        nonlocal get_new_obj

        print("Undoing last object placement...")
        if last_objects_positions == {}:
            print("No last object placement to undo.")
            return

        for i in range(env.unwrapped.tote_manager.num_objects):
            if i in last_objects_positions:
                asset = env.unwrapped.scene[f"object{i}"]
                pose = last_objects_positions[i]
                asset.write_root_pose_to_sim(pose.unsqueeze(0))

        last_objects_positions.clear()
        print("Last object placement undone.")

        get_new_obj = False

    def reset_target_cube() -> None:
        target.set_world_pose(np.array([0.5, -0.7, 0.2]), np.array([0, 0, 0, 1]))

    # Create device config if not already in env_cfg
    teleoperation_callbacks: dict[str, Callable[[], None]] = {
        "R": reset_recording_instance,
        "START": start_teleoperation,
        "STOP": stop_teleoperation,
        "RESET": reset_recording_instance,
        "ENTER": place_object,
        "NUMPAD_ENTER": place_object,
        "LEFT_SHIFT": undo_place,
        "TAB": reset_target_cube,
    }

    # Create teleop device from config if present, otherwise create manually
    teleop_interface = None
    try:
        if hasattr(env_cfg, "teleop_devices") and args_cli.teleop_device in env_cfg.teleop_devices.devices:
            teleop_interface = create_teleop_device(
                args_cli.teleop_device, env_cfg.teleop_devices.devices, teleoperation_callbacks
            )
        else:
            omni.log.warn(f"No teleop device '{args_cli.teleop_device}' found in environment config. Creating default.")
            # Create fallback teleop device
            sensitivity = args_cli.sensitivity
            if args_cli.teleop_device.lower() == "keyboard":
                teleop_interface = Se3Keyboard(
                    Se3KeyboardCfg(pos_sensitivity=0.01 * sensitivity, rot_sensitivity=0.5 * sensitivity)
                )
            elif args_cli.teleop_device.lower() == "spacemouse":
                teleop_interface = Se3SpaceMouse(
                    Se3SpaceMouseCfg(pos_sensitivity=0.05 * sensitivity, rot_sensitivity=0.05 * sensitivity)
                )
            else:
                omni.log.error(f"Unsupported teleop device: {args_cli.teleop_device}")
                omni.log.error("Supported devices: keyboard, spacemouse, gamepad, handtracking")
                env.close()
                simulation_app.close()
                return

            # Add callbacks to fallback device
            for key, callback in teleoperation_callbacks.items():
                try:
                    teleop_interface.add_callback(key, callback)
                except (ValueError, TypeError) as e:
                    omni.log.warn(f"Failed to add callback for key {key}: {e}")
    except Exception as e:
        omni.log.error(f"Failed to create teleop device: {e}")
        env.close()
        simulation_app.close()
        return

    if teleop_interface is None:
        omni.log.error("Failed to create teleop interface")
        env.close()
        simulation_app.close()
        return

    print(f"Using teleop device: {teleop_interface}")

    # reset environment
    obs = env.reset()[0]
    if "sensor" in obs:
        image_obs = obs["sensor"].permute(0, 3, 1, 2).flatten(start_dim=1)

    teleop_interface.reset()

    tote_ids = torch.zeros(env.unwrapped.num_envs, device=env.unwrapped.device).int()
    x_tote, y_tote, z_tote = env.unwrapped.tote_manager._tote_assets_state[tote_ids.item(), 0][0:3]
    x_tote_dim, y_tote_dim = (
        env.unwrapped.tote_manager.true_tote_dim[0] / 100,
        env.unwrapped.tote_manager.true_tote_dim[1] / 100,
    )

    # simulate environment
    while simulation_app.is_running():
        # run everything in inference mode
        with torch.inference_mode():
            # position and orientation of target virtual cube:
            cmd = teleop_interface.advance()

            if get_new_obj:
                tote_ids = torch.zeros(env.unwrapped.num_envs, device=env.unwrapped.device).int()
                packable_objects = env.unwrapped.bpp.get_packable_object_indices(
                    env.unwrapped.tote_manager.num_objects,
                    env.unwrapped.tote_manager,
                    torch.arange(env.unwrapped.num_envs, device=env.unwrapped.device),
                    tote_ids,
                )[0]
                # Update FIFO queues with new packable objects
                env.unwrapped.bpp.update_fifo_queues(packable_objects)

                # Select objects using FIFO (First In, First Out) ordering
                object_to_pack = env.unwrapped.bpp.select_fifo_packable_objects(packable_objects, env.unwrapped.device)
                asset = env.unwrapped.scene[f"object{object_to_pack.item()}"]
                asset.set_visibility(True, env_ids=torch.arange(env.unwrapped.num_envs, device=env.unwrapped.device))
            elif last_placed_object is not None:
                object_to_pack = torch.tensor(last_placed_object, device=env.unwrapped.device)
                last_placed_object = None

            cube_position, cube_orientation = target.get_world_pose()

            cube_position += cmd[:3].to(cube_position.device)
            cmd_rot = math_utils.quat_from_euler_xyz(cmd[3], cmd[4], cmd[5])
            cube_orientation = quat_mul(cmd_rot.to(cube_orientation.device), cube_orientation)
            # snap to increments of theta degrees
            cube_orientation = snap_to_theta_continuous(cube_orientation)

            rotated_dim = calculate_rotated_bounding_box(
                env.unwrapped.tote_manager.get_object_bbox(0, object_to_pack).unsqueeze(0),
                cube_orientation.unsqueeze(0),
                device=env.unwrapped.device,
            )

            cube_position[0] = max(
                x_tote - x_tote_dim / 2 + rotated_dim[0, 0] / 2,
                min(cube_position[0], x_tote + x_tote_dim / 2 - rotated_dim[0, 0] / 2),
            )
            cube_position[1] = max(
                y_tote - y_tote_dim / 2 + rotated_dim[0, 1] / 2,
                min(cube_position[1], y_tote + y_tote_dim / 2 - rotated_dim[0, 1] / 2),
            )

            x_cube, y_cube, z_cube = cube_position[0].item(), cube_position[1].item(), cube_position[2].item()

            x_diff = x_cube - x_tote + x_tote_dim / 2 - rotated_dim[0, 0] / 2
            y_diff = y_cube - y_tote + y_tote_dim / 2 - rotated_dim[0, 1] / 2

            if image_obs is None:
                raise ValueError("Depth image observation is None. Cannot compute z position.")
            z_pos = (
                get_z_position_from_depth(
                    image_obs, [x_diff, y_diff], rotated_dim, img_h, img_w, env.unwrapped.tote_manager.true_tote_dim
                )
                + 0.01
            )

            cube_position[2] = z_pos + rotated_dim[0, 2] / 2

            target.set_world_pose(cube_position, cube_orientation)

            target_pose = (cube_position, cube_orientation)
            asset = env.unwrapped.scene[f"object{object_to_pack.item()}"]
            asset.write_root_link_pose_to_sim(
                torch.cat([cube_position, cube_orientation], dim=0).unsqueeze(0),
            )
            asset.write_root_com_velocity_to_sim(torch.zeros(6, device=cube_position.device).unsqueeze(0))

            if past_pose is None:
                past_pose = (cube_position, cube_orientation)
            if target_pose is None:
                target_pose = (cube_position, cube_orientation)

            # compute zero actions
            past_pose = (cube_position, cube_orientation)
            env.unwrapped.scene.write_data_to_sim()
            env.unwrapped.sim.forward()
            env.unwrapped.sim.render()
            obs = env.unwrapped.observation_manager.compute()
            if "sensor" in obs:
                image_obs = obs["sensor"].permute(0, 3, 1, 2).flatten(start_dim=1)

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
