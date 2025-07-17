# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import time
from collections import defaultdict, deque
from typing import Any

import torch


class ToteStatistics:
    """
    Class for tracking and logging tote operation statistics.

    This tracks metrics like GCU over time, object transfers,
    and ejection counts for source and destination totes.
    """

    def __init__(self, num_envs: int, num_totes: int, device: torch.device):
        """
        Initialize the tote statistics tracker.

        Args:
            num_envs: Number of environments
            num_totes: Number of totes per environment
            device: Device for torch tensors
        """
        self.num_envs = num_envs
        self.num_totes = num_totes
        self.device = device

        # Initialize statistics trackers
        self.step_count = 0

        # GCU history per environment and tote
        self.gcu_history = []

        # Track operations with step information
        self.operations = defaultdict(list)

        # Object transfer counts per step and environment
        self.obj_transfers_history = []
        self.obj_transfers = torch.zeros(num_envs, device=device)

        # Ejection counts per step and environment
        self.source_tote_ejections_history = []
        self.dest_tote_ejections_history = []
        self.source_tote_ejections = torch.zeros(num_envs, device=device)
        self.dest_tote_ejections = torch.zeros(num_envs, device=device)

        # Inbound and outbound GCU values for each tote
        self.inbound_gcus = [[deque() for _ in range(num_totes)] for _ in range(num_envs)]
        self.outbound_gcus = [[deque() for _ in range(num_totes)] for _ in range(num_envs)]

        # Comprehensive history at destination tote ejections
        self.ejection_snapshots = [[] for _ in range(self.num_envs)]

        # Timestamp when logging started
        self.start_time = time.time()

    def log_gcu(self, gcu_values: torch.Tensor, env_ids: torch.Tensor = None):
        """
        Log GCU values for specified environments.

        Args:
            gcu_values: Tensor of GCU values [num_envs, num_totes]
            env_ids: Optional tensor of environment IDs to log for
        """
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)

        # Store the GCU values with step number
        self.gcu_history.append({
            "step": self.step_count,
            "values": gcu_values.detach().clone(),
            "env_ids": env_ids.detach().clone(),
            "timestamp": time.time() - self.start_time,
        })

    def log_obj_transfers(self, env_ids: torch.Tensor, num_objects: int = 1):
        """
        Log object transfer operations.

        Args:
            env_ids: Tensor of environment IDs where transfers occurred
            num_objects: Number of objects placed/transferred
        """
        self.obj_transfers[env_ids] += num_objects

        # Track operation with step information
        self.operations["object_transfers"].append({
            "step": self.step_count,
            "env_ids": env_ids.detach().clone().cpu().numpy().tolist(),
            "count": num_objects,
            "timestamp": time.time() - self.start_time,
        })

        # Record the current object transfer state for all environments
        self.obj_transfers_history.append({
            "step": self.step_count,
            "values": self.obj_transfers.detach().clone(),
            "timestamp": time.time() - self.start_time,
        })

    def log_tote_eject_gcus(self, inbound_gcus: torch.Tensor, outbound_gcus: torch.Tensor, totes_ejected: torch.Tensor):
        """
        Log GCU values for totes that were ejected.
        Args:
            inbound_gcus: Tensor of GCU values for inbound totes [num_envs, num_totes]
            outbound_gcus: Tensor of GCU values for outbound totes   [num_envs, num_totes]
            totes_ejected: Boolean tensor indicating which totes were ejected [num_envs, num_totes]
        """
        ejected_indices = totes_ejected.nonzero(as_tuple=False)  # Shape: [N, 2]

        for env, tote in ejected_indices:
            self.inbound_gcus[env][tote].append(inbound_gcus[env, tote].item())
            self.outbound_gcus[env][tote].append(outbound_gcus[env, tote].item())

    def log_source_tote_ejection(self, env_ids: torch.Tensor):
        """
        Log source tote ejection events.

        Args:
            env_ids: Tensor of environment IDs where ejections occurred
        """
        if self.step_count == 0:
            return  # Skip logging on the first step
        self.source_tote_ejections[env_ids] += 1
        count = len(env_ids)

        # Track operation with step information
        self.operations["source_tote_ejection"].append({
            "step": self.step_count,
            "env_ids": env_ids.detach().clone().cpu().numpy().tolist(),
            "count": count,
            "timestamp": time.time() - self.start_time,
        })

        # Record the current ejection state for all environments
        self.source_tote_ejections_history.append({
            "step": self.step_count,
            "values": self.source_tote_ejections.detach().clone(),
            "timestamp": time.time() - self.start_time,
        })

    def log_dest_tote_ejection(self, tote_ids: torch.Tensor, env_ids: torch.Tensor):
        """
        Log destination tote ejection events.

        Args:
            tote_ids: Tensor of tote IDs where ejections occurred
            env_ids: Tensor of environment IDs where ejections occurred
        """
        self.dest_tote_ejections[env_ids] += 1
        count = len(env_ids)
        for idx, env_id in enumerate(env_ids):
            self.ejection_snapshots[env_id].append([tote_ids[idx], self.get_summary(env_id)])
            # Reset counts
            self.source_tote_ejections[env_id] = 0
            self.dest_tote_ejections[env_id] = 0
            self.obj_transfers[env_id] = 0

        # Track operation with step information
        self.operations["dest_tote_ejection"].append({
            "step": self.step_count,
            "env_ids": env_ids.detach().clone().cpu().numpy().tolist(),
            "count": count,
            "timestamp": time.time() - self.start_time,
        })

        # Record the current ejection state for all environments
        self.dest_tote_ejections_history.append({
            "step": self.step_count,
            "values": self.dest_tote_ejections.detach().clone(),
            "timestamp": time.time() - self.start_time,
        })

    def increment_step(self):
        """Increment the step counter."""
        self.step_count += 1

    def get_summary(self, env_ids: torch.Tensor | int | None = None) -> dict[str, Any]:
        """
        Get a summary of statistics for specified environment IDs.

        Args:
            env_ids: Environment IDs to include in the summary. Can be a tensor, single integer, or None (for all environments)

        Returns:
            Dictionary containing summarized statistics and history for specified environments
        """
        # Handle different input types for env_ids
        if isinstance(env_ids, int):
            env_ids = torch.tensor([env_ids], device=self.device)
        elif env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)

        # Convert env_ids to list for easier filtering
        env_id_list = env_ids.cpu().numpy().tolist()
        if not isinstance(env_id_list, list):
            env_id_list = [env_id_list]

        # Filter operations by environment IDs
        filtered_operations = defaultdict(list)
        for op_name, op_list in self.operations.items():
            for entry in op_list:
                entry_env_ids = entry["env_ids"] if isinstance(entry["env_ids"], list) else [entry["env_ids"]]
                if any(env_id in env_id_list for env_id in entry_env_ids):
                    filtered_operations[op_name].append(entry)

        # Create filtered history entries
        def filter_history(history_list):
            return [
                {
                    "step": entry["step"],
                    "values": entry["values"][env_ids].cpu().numpy().tolist(),
                    "timestamp": entry["timestamp"],
                }
                for entry in history_list
            ]

        # Filter GCU history
        filtered_gcu_history = []
        for entry in self.gcu_history:
            entry_env_ids = entry["env_ids"]
            mask = torch.zeros_like(entry_env_ids, dtype=torch.bool)

            # Create mask for matching environment IDs
            if env_ids.ndim == 0:
                mask |= entry_env_ids == env_ids.item()
            else:
                for env_id in env_ids:
                    mask |= entry_env_ids == env_id

            if mask.any():
                filtered_indices = torch.nonzero(mask, as_tuple=True)[0]
                filtered_entry_env_ids = entry_env_ids[filtered_indices]

                filtered_gcu_history.append({
                    "step": entry["step"],
                    "values": entry["values"][filtered_entry_env_ids].cpu().numpy().tolist(),
                    "env_ids": filtered_entry_env_ids.cpu().numpy().tolist(),
                    "timestamp": entry["timestamp"],
                })

        return {
            "total_steps": self.step_count,
            "operations_history": dict(filtered_operations),
            "current_object_transfers": self.obj_transfers[env_ids].cpu().numpy().tolist(),
            "current_source_tote_ejections": self.source_tote_ejections[env_ids].cpu().numpy().tolist(),
            "current_dest_tote_ejections": self.dest_tote_ejections[env_ids].cpu().numpy().tolist(),
            "ejection_snapshots": [self.ejection_snapshots[i] for i in env_id_list if i < len(self.ejection_snapshots)],
            "source_tote_ejections_history": filter_history(self.source_tote_ejections_history),
            "dest_tote_ejections_history": filter_history(self.dest_tote_ejections_history),
            "obj_transfers_history": filter_history(self.obj_transfers_history),
            "gcu_history": filtered_gcu_history,
        }

    def get_ejection_summary(self) -> dict[str, Any]:
        """
        Get a summary of the most recent ejection statistics.

        Returns:
            Dictionary containing the most recent ejection data
        """
        summary = {}
        # Check if there are any snapshots
        for env_id in range(self.num_envs):
            if self.ejection_snapshots[env_id]:
                # Return a summary of the last snapshot for this environment
                tote_ids = self.ejection_snapshots[env_id][-1][0]
                summary_env = self.ejection_snapshots[env_id][-1][1]
                last_gcu = (
                    summary_env["gcu_history"][-1]["values"][0][tote_ids]
                    if "gcu_history" in summary_env and len(summary_env["gcu_history"]) > 0
                    else None
                )
                last_obj_transfers = (
                    summary_env["obj_transfers_history"][-1]["values"]
                    if "obj_transfers_history" in summary_env and len(summary_env["obj_transfers_history"]) > 0
                    else None
                )
                last_source_ejection = (
                    summary_env["source_tote_ejections_history"][-1]["values"]
                    if "source_tote_ejections_history" in summary_env
                    and len(summary_env["source_tote_ejections_history"]) > 0
                    else None
                )
                summary[env_id] = {}
                summary[env_id]["last_gcu"] = last_gcu
                summary[env_id]["last_source_ejection"] = last_source_ejection
                summary[env_id]["last_obj_transfers"] = last_obj_transfers
        return summary

    def get_full_ejection_summary(self) -> dict[str, Any]:
        """
        Get a comprehensive summary of all ejection statistics.
        """
        summary = {}
        mean_gcus = 0
        mean_obj_transfers = 0
        mean_source_ejections = 0
        data_count = 0
        for env_id in range(self.num_envs):
            if self.ejection_snapshots[env_id] is not None and len(self.ejection_snapshots[env_id]) > 0:
                summary[env_id] = {"gcus": [], "obj_transfers": [], "source_ejections": []}
                for i in range(len(self.ejection_snapshots[env_id])):
                    tote_ids = self.ejection_snapshots[env_id][i][0]
                    summary_env = self.ejection_snapshots[env_id][i][1]
                    last_gcu = (
                        summary_env["gcu_history"][-1]["values"][0][tote_ids]
                        if "gcu_history" in summary_env and len(summary_env["gcu_history"]) > 0
                        else None
                    )
                    last_obj_transfers = (
                        summary_env["obj_transfers_history"][-1]["values"]
                        if "obj_transfers_history" in summary_env and len(summary_env["obj_transfers_history"]) > 0
                        else 0
                    )
                    last_source_ejection = (
                        summary_env["source_tote_ejections_history"][-1]["values"]
                        if "source_tote_ejections_history" in summary_env
                        and len(summary_env["source_tote_ejections_history"]) > 0
                        else 0
                    )
                    if last_gcu is not None:
                        mean_gcus += last_gcu
                    mean_obj_transfers += last_obj_transfers
                    mean_source_ejections += last_source_ejection
                    data_count += 1
                    summary[env_id]["gcus"].append(last_gcu)
                    summary[env_id]["obj_transfers"].append(last_obj_transfers)
                    summary[env_id]["source_ejections"].append(last_source_ejection)
        summary["mean_gcus"] = mean_gcus / data_count if data_count > 0 else 0
        summary["mean_obj_transfers"] = mean_obj_transfers / data_count if data_count > 0 else 0
        summary["mean_source_ejections"] = mean_source_ejections / data_count if data_count > 0 else 0
        return summary

    def save_to_file(self, filepath: str):
        """
        Save statistics to a file.

        Args:
            filepath: Path to save the statistics
        """
        import json

        # Convert data to JSON-serializable format
        summary = self.get_full_ejection_summary()

        with open(filepath, "w") as f:
            json.dump(summary, f, indent=4)
