import math
import time

import isaaclab.utils.math as math_utils
import matplotlib.pyplot as plt
import numpy as np
import torch
from packing3d import (
    Attitude,
    Container,
    Display,
    Item,
    PackingProblem,
    Position,
    Transform,
)
from tote_consolidation.tasks.manager_based.pack.utils.tote_helpers import (
    calculate_rotated_bounding_box,
)


class BPP:
    def __init__(self, tote_manager, objects, scale=1.0):
        """
        Initialize the BPP utility class.

        Args:
            scale (float): Scale factor for dimensions
        """
        self.scale = scale
        self.objects = objects
        self.tote_manager = tote_manager
        self.tote_dims, self.obj_dims, self.obj_voxels = self.get_packing_variables()
        self.packed_obj_idx = []
        self.unpackable_obj_idx = []  # Note: Currently, no reconsideration of unpackable objects
        self.source_eject_tries = 0
        self.max_source_eject_tries = tote_manager.num_totes - 1  # Maximum number of ejects allowed
        items = []
        for i in range(len(self.obj_dims)):
            if i in objects:
                # Create an item for each packable object
                items.append(Item(np.array(self.obj_voxels[0][i], dtype=np.float32)))

        # x y z to z x y
        self.display = Display(self.tote_dims)

        self.problem = PackingProblem(self.tote_dims, items)

    def get_packing_variables(self):
        """
        Extract and format packing variables from the tote manager.

        Args:
            tote_manager: The tote manager containing environment information

        Returns:
            tuple: Tote dimensions, object dimensions, and object voxels
        """
        tote_dims = self.tote_manager.true_tote_dim.tolist()
        tote_dims = [int(tote_dims[2] * self.scale), int(tote_dims[0] * self.scale), int(tote_dims[1] * self.scale)]

        obj_dims = self.tote_manager.obj_bboxes.tolist()[0]  # Only first environment for now
        obj_dims = [[int(dim * self.scale) for dim in dims] for dims in obj_dims]

        obj_voxels = self.tote_manager.obj_voxels
        # TODO (kaikwan): Scale voxel grid as well
        if self.scale != 1.0:
            # Warn if scaling is applied
            print(
                "Warning: Scaling is applied to tote dimensions but not to voxel grid. This may lead to inaccuracies in"
                " packing."
            )
        return tote_dims, obj_dims, obj_voxels

    def update_container_heightmap(self, env, env_indices, tote_ids):
        """
        Update the heightmap of the container in the packing problem with updated
        object poses in sim.

        Args:
            problem: The packing problem containing the container and items
            tote_manager: The sim tote manager containing the current state of the environment
            env_indices: Indices of the environments being updated
            tote_ids: IDs of the destination totes for each environment

        Returns:
            PackingProblem: Updated packing problem
        """
        # TODO (kaikwan): Potential optimization: Only update the top most objects in the container
        # This is because the container heightmap is only used for the top most objects
        # and the rest of the objects are already packed in the container and unlikely to change or affect planning
        t = time.time()
        # Clear the current container geometry
        self.problem.container = Container(self.problem.box_size)

        # For each object in dest tote, get the current transform and add it to the container
        for env_idx in env_indices:
            tote_id = tote_ids[env_idx].item()
            if tote_id < 0:
                continue
            for obj_idx in self.packed_obj_idx:
                obj_idx = obj_idx.item()
                asset_name = "object" + str(obj_idx)
                asset = env.unwrapped.scene[asset_name]
                asset_pos = asset.data.root_state_w[env_idx, :3]
                asset_quat = asset.data.root_state_w[env_idx, 3:7]

                quat_init = torch.tensor([1, 0, 0, 0], device=env.unwrapped.device)
                asset_quat = math_utils.quat_mul(math_utils.quat_inv(quat_init), asset_quat)

                bbox_offset = self.tote_manager.obj_bboxes[env_idx, obj_idx]
                rotated_half_dim = (
                    calculate_rotated_bounding_box(
                        bbox_offset.unsqueeze(0), asset_quat.unsqueeze(0), device=env.unwrapped.device
                    )
                    / 2.0
                ).squeeze(0)
                asset_pos = asset_pos - rotated_half_dim

                asset_pos[0] += self.tote_manager.true_tote_dim[0] / 2 * 0.01  # Adjust for center of tote
                asset_pos[1] += self.tote_manager.true_tote_dim[1] / 2 * 0.01  # Adjust for center of tote

                asset_pos = asset_pos - self.tote_manager._tote_assets_state.permute(1, 0, 2)[env_idx, tote_id]

                asset_pos = asset_pos * 100  # convert to cm resolution

                euler_angles = math_utils.euler_xyz_from_quat(asset_quat.unsqueeze(0))
                # Convert tuple to tensor if it's a tuple
                if isinstance(euler_angles, tuple):
                    euler_angles = torch.tensor(euler_angles, device=env.unwrapped.device)
                # radians to degrees
                euler_angles = euler_angles.squeeze(0) * 180 / np.pi
                curr_item = self.problem.items[obj_idx]
                curr_item.transform(
                    Transform(
                        position=Position(
                            max(0, math.floor(asset_pos[0].item())),
                            max(0, math.floor(asset_pos[1].item())),
                            max(0, math.floor(asset_pos[2].item())),
                        ),
                        attitude=Attitude(
                            roll=euler_angles[0].item(), pitch=euler_angles[1].item(), yaw=euler_angles[2].item()
                        ),
                    )
                )
                self.problem.container.add_item(curr_item)
        print("Time taken to update container heightmap: ", time.time() - t)

    def get_action(self, env, obj_indicies, tote_ids):
        """
        Get the action for the current step in the packing problem.

        Args:
            obj_indicies: Indices of objects to consider

        Returns:
            tuple: Updated problem, transform, and object index
        """

        tote_id = tote_ids[0]  # first env dest tote
        source_tote_ids = [i for i in range(self.tote_manager.num_totes) if i != tote_id]
        env_idx = torch.tensor([0], device=env.unwrapped.device)

        # If all max tries are reached or no objects in reserve, eject and load destination tote
        # TODO (kaikwan): Should think more carefully about this
        # Take into consideration of fullness/GCU to decide whether to eject source tote or destination tote
        if (
            self.source_eject_tries >= self.max_source_eject_tries
            or self.tote_manager.get_reserved_objs_idx(env_idx).sum() == 0
        ):
            print("No objects in reserve, force ejecting and loading destination tote.")
            dest_tote_tensor = torch.tensor([tote_id], device=env.unwrapped.device)

            self.tote_manager.eject_totes(env.unwrapped, dest_tote_tensor, env_idx, is_dest=True, overfill_check=False)
            new_packable_objects, _ = self.get_packable_object_indices(
                self.tote_manager.num_objects, self.tote_manager, env_idx, dest_tote_tensor
            )
            # Reset variable for unpackable objects and container
            self.unpackable_obj_idx = []
            self.source_eject_tries = 0
            self.problem.container = Container(self.problem.box_size)
            return self.get_action(env, new_packable_objects, tote_ids)
        # If all current objects are unpackable, force eject and load source tote
        elif len(obj_indicies[env_idx]) == 0:
            print("All objects are unpackable, force ejecting and loading source tote.")

            source_tote_tensor = torch.tensor(source_tote_ids, device=env.unwrapped.device)[
                self.source_eject_tries
            ].unsqueeze(0)

            # Remove objects from unpackable list
            self.unpackable_obj_idx = [
                idx
                for idx in self.unpackable_obj_idx
                if idx not in self.tote_manager.get_tote_objs_idx(tote_id, env_idx)
            ]

            self.tote_manager.eject_totes(
                env.unwrapped, source_tote_tensor, env_idx, is_dest=False, overfill_check=False
            )
            new_packable_objects, _ = self.get_packable_object_indices(
                self.tote_manager.num_objects, self.tote_manager, env_idx, tote_id
            )

            # Remove previously unpackable objects from the list
            new_packable_objects[env_idx] = [
                idx for idx in new_packable_objects[env_idx] if idx not in self.unpackable_obj_idx
            ]

            self.source_eject_tries += 1
            return self.get_action(env, new_packable_objects, tote_ids)

        t = time.time()
        obj_idx = obj_indicies[torch.randint(0, len(obj_indicies), (1,))][0]
        curr_item = self.problem.items[obj_idx.item()]
        obj_idx = obj_idx.unsqueeze(0)
        transforms = self.problem.container.search_possible_position(curr_item, step_width=90)
        if transforms:
            transform = transforms[0]
            curr_item.transform(transform)
            self.problem.container.add_item(curr_item)
            print("Time taken to find placement pose: ", time.time() - t)
            self.display.show3d(self.problem.container.geometry)
            self.packed_obj_idx.append(obj_idx)
            plt.savefig("packed_container.png")  # Save the figure
            return transform, obj_idx
        # No valid position found
        print("Unpackable object found, adding to unpackable list. Trying other object in existing source totes.")
        self.unpackable_obj_idx.append(obj_idx)
        # Remove unpackable object from list and try again
        obj_indicies[env_idx] = [idx for idx in obj_indicies[env_idx] if idx != obj_idx]

        return self.get_action(env, obj_indicies, tote_ids)

    def get_packable_object_indices(self, num_obj_per_env, tote_manager, env_indices, tote_ids):
        """Get indices of objects that can be packed per environment.

        Args:
            num_obj_per_env: Number of objects per environment
            tote_manager: The tote manager object
            env_indices: Indices of environments to get packable objects for
            tote_ids: Destination tote IDs for each environment

        Returns:
            List of tensors containing packable object indices for each environment
        """
        num_envs = env_indices.shape[0]

        # Get objects that are reserved (already being picked up)
        reserved_objs = tote_manager.get_reserved_objs_idx(env_indices)

        # Get objects that are already in destination totes
        objs_in_dest = tote_manager.get_tote_objs_idx(tote_ids, env_indices)

        # Create a 2D tensor of object indices: shape (num_envs, num_obj_per_env)
        obj_indices = torch.arange(0, num_obj_per_env, device=env_indices.device).expand(num_envs, -1)

        # Compute mask of packable objects
        mask = (~reserved_objs & ~objs_in_dest).bool()

        # Use list comprehension to get valid indices per environment
        valid_indices = [obj_indices[i][mask[i]] for i in range(num_envs)]

        return valid_indices, mask
