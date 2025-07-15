import math

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
        return tote_dims, obj_dims, obj_voxels

    def pack_item(self, item_idx, transform):
        """
        Pack an item into the container.

        Args:
            item_idx: Index of the item to pack
            transform: Transform to apply to the item
        """
        curr_item: Item = self.problem.items[item_idx]
        curr_item.transform(transform)
        self.problem.container.add_item(curr_item)

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

                asset_pos = asset_pos - torch.tensor([0, 0, 0.02], device=env.unwrapped.device)

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

    def get_action(self, obj_indicies):
        """
        Get the action for the current step in the packing problem.

        Args:
            obj_indicies: Indices of objects to consider

        Returns:
            tuple: Updated problem, transform, and object index
        """
        obj_idx = obj_indicies[torch.randint(0, len(obj_indicies), (1,))][0]
        import time

        t = time.time()
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
        return None  # No valid position found
