import math
import time

import isaaclab.utils.math as math_utils
import matplotlib

matplotlib.use("Agg")  # Use non-interactive backend for saving figures
import itertools
from concurrent.futures import ProcessPoolExecutor, as_completed
from time import perf_counter

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
    calculate_rotated_bounding_box_np,
)


class BPP:
    def __init__(self, tote_manager, num_envs, objects, scale=1.0):
        """
        Initialize the BPP utility class.

        Args:
            scale (float): Scale factor for dimensions
        """
        self.scale = scale
        self.objects = objects
        self.tote_manager = tote_manager
        self.tote_dims, self.obj_dims, self.obj_voxels = self.get_packing_variables()
        self.packed_obj_idx = [[] for _ in range(num_envs)]  # Packed objects per environment
        self.unpackable_obj_idx = [[] for _ in range(num_envs)]  # Unpackable objects per environment
        self.source_eject_tries = [0] * num_envs  # Number of tries to eject source totes
        self.max_source_eject_tries = 3  # Maximum number of ejects allowed
        self.num_envs = num_envs
        items = [[Item(np.array(self.obj_voxels[i][j], dtype=np.float32)) for j in objects] for i in range(num_envs)]

        # # x y z to z x y
        self.display = Display(self.tote_dims)

        self.problems = [PackingProblem(self.tote_dims, items[i]) for i in range(num_envs)]

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

        # obj_dims = self.tote_manager.obj_bboxes.tolist()[0]  # Only first environment for now
        # obj_dims = [[int(dim * self.scale) for dim in dims] for dims in obj_dims]

        obj_dims = self.tote_manager.obj_bboxes * self.scale  # Scale the bounding boxes
        obj_dims = obj_dims.to(dtype=torch.int32).tolist()

        obj_voxels = self.tote_manager.obj_voxels
        # TODO (kaikwan): Scale voxel grid as well
        if self.scale != 1.0:
            # Warn if scaling is applied
            print(
                "Warning: Scaling is applied to tote dimensions but not to voxel grid. This may lead to inaccuracies in"
                " packing."
            )
        return tote_dims, obj_dims, obj_voxels

    @staticmethod
    def update_container_worker_static(args):
        """
        Static worker for multiprocessing. Now also applies the transforms to the container.
        Args is a dict with keys: env_idx, tote_id, objects (list of dicts), problem, items.
        Returns a tuple (env_idx, results, container_items) where container_items is a list of (obj_idx, transform) that were added to the container.
        """
        env_idx = args["env_idx"]
        tote_id = args["tote_id"]
        objects = args["objects"]
        problem = args.get("problem")
        items = args.get("items")
        results = []
        container_items = []
        if tote_id < 0:
            return (env_idx, results, container_items)
        if problem is not None and items is not None:
            problem.container.clear()
        for obj in objects:
            asset_pos = torch.tensor(obj["asset_pos"])
            asset_quat = torch.tensor(obj["asset_quat"])
            bbox_offset = torch.tensor(obj["bbox_offset"])
            true_tote_dim = torch.tensor(obj["true_tote_dim"])
            tote_assets_state = torch.tensor(obj["tote_assets_state"])
            quat_init = torch.tensor([1, 0, 0, 0])
            asset_quat = math_utils.quat_mul(math_utils.quat_inv(quat_init), asset_quat)
            rotated_half_dim = (
                calculate_rotated_bounding_box_np(bbox_offset.unsqueeze(0), asset_quat.unsqueeze(0), device="cpu") / 2.0
            ).squeeze(0)
            asset_pos = asset_pos - rotated_half_dim
            asset_pos[0] += true_tote_dim[0] / 2 * 0.01
            asset_pos[1] += true_tote_dim[1] / 2 * 0.01
            asset_pos = asset_pos - tote_assets_state
            asset_pos = asset_pos * 100  # convert to cm resolution
            euler_angles = math_utils.euler_xyz_from_quat(asset_quat.unsqueeze(0))
            if isinstance(euler_angles, tuple):
                euler_angles = torch.tensor(euler_angles)
            euler_angles = euler_angles.squeeze(0) * 180 / np.pi
            transform = Transform(
                position=Position(
                    max(0, math.floor(asset_pos[0].item())),
                    max(0, math.floor(asset_pos[1].item())),
                    max(0, math.floor(asset_pos[2].item())),
                ),
                attitude=Attitude(
                    roll=euler_angles[0].item(), pitch=euler_angles[1].item(), yaw=euler_angles[2].item()
                ),
            )
            results.append((obj["obj_idx"], transform))
            if problem is not None and items is not None:
                curr_item = items[obj["obj_idx"]]
                curr_item.transform(transform)
                problem.container.add_item(curr_item)
                container_items.append((obj["obj_idx"], transform))
        return (env_idx, results, container_items, problem.container)

    def _extract_env_data(self, env, env_idx, tote_id):
        """
        Extract all necessary data from env for a single environment index.
        Returns a list of dicts, one per packed object in this env.
        Prints device/type info for debugging CUDA issues.
        """
        data = []
        for obj_idx in self.packed_obj_idx[env_idx]:
            obj_idx_val = obj_idx.item()
            asset_name = "object" + str(obj_idx_val)
            asset = env.unwrapped.scene[asset_name]
            asset_pos = asset.data.root_state_w[env_idx, :3].detach().cpu().numpy()
            asset_quat = asset.data.root_state_w[env_idx, 3:7].detach().cpu().numpy()
            bbox_offset = self.tote_manager.obj_bboxes[env_idx, obj_idx_val].detach().cpu().numpy()
            true_tote_dim = self.tote_manager.true_tote_dim.detach().cpu().numpy()
            tote_assets_state = self.tote_manager._tote_assets_state.permute(1, 0, 2).detach().cpu().numpy()
            data.append({
                "obj_idx": obj_idx_val,
                "asset_pos": asset_pos,
                "asset_quat": asset_quat,
                "bbox_offset": bbox_offset,
                "true_tote_dim": true_tote_dim,
                "tote_assets_state": tote_assets_state[env_idx, tote_id],
            })
        return data

    def update_container_heightmap(self, env, env_indices, tote_ids):
        """
        Update the heightmap of the container in the packing problem with updated
        object poses in sim, using batched multiprocessing per environment.
        """
        t = time.time()
        batch_args = []
        for env_idx in env_indices:
            tote_id = tote_ids[env_idx].item()
            if tote_id < 0:
                continue
            objects = self._extract_env_data(env, env_idx, tote_id)
            batch_args.append({
                "env_idx": env_idx.detach().cpu().item(),
                "tote_id": tote_id,
                "objects": objects,
                "problem": self.problems[env_idx],
                "items": self.problems[env_idx].items,
            })
        results = []
        with ProcessPoolExecutor(max_workers=20) as executor:
            for res in executor.map(BPP.update_container_worker_static, batch_args):
                results.append(res)
        for env_idx, res, container_items, container in results:
            self.problems[env_idx].container = container
        print("Time taken for updating container heightmap: ", time.time() - t)

    @staticmethod
    def _env_worker(args):
        i, env_idx, obj_indices, problem, plot = args
        if len(obj_indices) == 0:
            return (i, (None, None, None))

        item = problem.items[obj_indices[0]]
        obj_idx = obj_indices[0].item()
        transforms = problem.container.search_possible_position(item, grid_num=30, step_width=90)
        if len(transforms) != 0:
            return (i, (obj_idx, torch.tensor([obj_idx]), transforms[0]))
        # For stability check transforms
        # for transform in transforms:
        #     test_item = deepcopy(item)
        #     test_item.transform(transform)
        #     if container.check_stability_with_candidate(test_item, plot=plot):
        #         return (i, (obj_idx, torch.tensor([obj_idx]), transform))
        return (i, (None, torch.tensor([obj_idx]), None))

    def _eject_and_reload(self, env_idx, tote_ids, is_dest, overfill_check=False):
        if is_dest:
            tote_tensor = tote_ids[env_idx].unsqueeze(0)
        else:
            tote_tensor = tote_ids[env_idx][self.source_eject_tries[env_idx] % len(tote_ids[env_idx])].unsqueeze(0)
        if not is_dest:
            tote_objects = self.tote_manager.get_tote_objs_idx(tote_tensor, env_idx.unsqueeze(0))
            objects_to_remove = torch.where(tote_objects == 1)[1]
            self.unpackable_obj_idx[env_idx] = [
                idx for idx in self.unpackable_obj_idx[env_idx] if idx not in objects_to_remove
            ]
        self.tote_manager.eject_totes(tote_tensor, env_idx.unsqueeze(0), is_dest=is_dest, overfill_check=overfill_check)

        if not is_dest:
            self.tote_manager.refill_source_totes(env_idx.unsqueeze(0))
            self.source_eject_tries[env_idx] += 1
        else:
            self.problems[env_idx].container = Container(self.tote_dims)
            self.packed_obj_idx[env_idx] = []
            self.unpackable_obj_idx[env_idx] = []
            self.source_eject_tries[env_idx] = 0

    def _get_new_objs(self, env_idx, dest_tote_ids):
        new_packable_objects, _ = self.get_packable_object_indices(
            self.tote_manager.num_objects, self.tote_manager, env_idx.unsqueeze(0), dest_tote_ids[env_idx].unsqueeze(0)
        )
        return [new_packable_objects[0][i].cpu() for i in range(len(new_packable_objects[0]))]

    def get_action(self, env, obj_indices, dest_tote_ids, env_indices, plot=False):
        t = time.time()
        source_tote_ids = torch.arange(0, self.tote_manager.num_totes, device=env.unwrapped.device).repeat(
            self.num_envs, 1
        )
        source_tote_ids = source_tote_ids[source_tote_ids != dest_tote_ids.unsqueeze(1)].reshape(self.num_envs, -1)

        transforms_list = [None] * len(env_indices)
        obj_idx_list = [None] * len(env_indices)

        obj_indices = [[obj_indices[i][j].cpu() for j in range(len(obj_indices[i]))] for i in range(len(env_indices))]
        pending_envs = [(i, env_idx, obj_indices[env_idx]) for i, env_idx in enumerate(env_indices.cpu())]

        # Global parallel pool
        with ProcessPoolExecutor(max_workers=20) as executor:
            futures = {}

            # Submit initial jobs
            for i, env_idx, curr_obj_indices in pending_envs:
                job_args = (i, env_idx.cpu(), curr_obj_indices, self.problems[env_idx], plot)
                futures[executor.submit(self._env_worker, job_args)] = (i, env_idx, curr_obj_indices)

            while futures:
                for future in as_completed(futures):
                    i, env_idx, curr_obj_indices = futures.pop(future)
                    env_idx = env_idx.to(env.unwrapped.device)

                    _, (obj_idx, obj_idx_tensor, transform) = future.result()
                    curr_obj_indices = [obj for obj in curr_obj_indices if obj not in self.unpackable_obj_idx[env_idx]]

                    # ---- fallback logic as before ----
                    if (
                        self.source_eject_tries[env_idx] >= self.max_source_eject_tries
                        or self.tote_manager.get_reserved_objs_idx(env_idx.unsqueeze(0)).sum() == 0
                    ):
                        if self.source_eject_tries[env_idx] >= self.max_source_eject_tries:
                            print(
                                f"Max source eject tries {self.source_eject_tries[env_idx]} reached for environment"
                                f" {env_idx}. Ejecting destination tote."
                            )
                        else:
                            print(f"No reserved objects in environment {env_idx}. Ejecting destination tote.")
                        self._eject_and_reload(env_idx, dest_tote_ids, is_dest=True, overfill_check=False)
                        new_objs = self._get_new_objs(env_idx, dest_tote_ids)
                        job_args = (i, env_idx.cpu(), new_objs, self.problems[env_idx], plot)
                        futures[executor.submit(self._env_worker, job_args)] = (i, env_idx.cpu(), new_objs)
                        continue
                    elif len(curr_obj_indices) == 0:
                        print(
                            "No packable objects left in environment {}. Ejecting source tote despite not empty."
                            .format(env_idx)
                        )
                        self._eject_and_reload(env_idx, source_tote_ids, is_dest=False)
                        new_objs = self._get_new_objs(env_idx, dest_tote_ids)
                        job_args = (i, env_idx.cpu(), new_objs, self.problems[env_idx], plot)
                        futures[executor.submit(self._env_worker, job_args)] = (i, env_idx.cpu(), new_objs)
                        continue
                    elif obj_idx is not None:
                        transforms_list[i] = transform
                        obj_idx_list[i] = obj_idx_tensor
                        self.packed_obj_idx[env_idx].append(obj_idx_tensor)
                        # curr_item = self.problems[env_idx].items[obj_idx_tensor.item()]
                        # curr_item.transform(transform)
                        # self.problems[env_idx].container.add_item(curr_item)
                        # self.display.show3d(self.problems[env_idx].container.geometry)
                        # plt.savefig(f"packed_container_{env_idx}.png")  # Save the figure
                        print(f"Placement Time taken for environment {env_idx}: {time.time() - t}")
                    elif obj_idx_tensor is not None:
                        print(
                            "No valid transform found for object index {} in environment {}. Adding to unpackable"
                            " objects.".format(obj_idx_tensor, env_idx)
                        )
                        self.unpackable_obj_idx[env_idx].extend(obj_idx_tensor)
                        remaining_objs = [
                            obj for obj in curr_obj_indices if obj not in self.unpackable_obj_idx[env_idx]
                        ]
                        job_args = (i, env_idx.cpu(), remaining_objs, self.problems[env_idx], plot)
                        futures[executor.submit(self._env_worker, job_args)] = (i, env_idx.cpu(), remaining_objs)
                        continue
                    else:
                        raise ValueError(f"No valid object index or transform found for environment {env_idx}")
        valid_transforms = [t for t in transforms_list if t is not None]
        valid_obj_idxs = [o for o in obj_idx_list if o is not None]

        if not valid_transforms:
            raise ValueError("No valid transforms found for any environment.")
        if len(valid_obj_idxs) != self.num_envs:
            raise ValueError("Not all environments have valid object indices.")

        return valid_transforms, torch.cat(valid_obj_idxs, dim=0) if valid_obj_idxs else None

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

        # Remove unpackable objects from valid indices
        for i in range(num_envs):
            valid_indices[i] = [obj for obj in valid_indices[i] if obj not in self.unpackable_obj_idx[i]]

        return valid_indices, mask
