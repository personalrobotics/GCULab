# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import logging
import time
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import torch

logger = logging.getLogger(__name__)


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

        # Log to console for visibility
        mean_gcu = gcu_values.mean().item()
        max_gcu = gcu_values.max().item()
        logger.info(f"Step {self.step_count} - Mean GCU: {mean_gcu:.4f}, Max GCU: {max_gcu:.4f}")

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

        logger.debug(f"Object transfer: {num_objects} objects")

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

        logger.info(f"Source tote ejection in {count} environments")

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
            self.ejection_snapshots[env_id].append([tote_ids[idx], self.get_summary()])
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

        logger.info(f"Destination tote ejection in {count} environments")

    def increment_step(self):
        """Increment the step counter."""
        self.step_count += 1

    def get_summary(self) -> dict[str, Any]:
        """
        Get a summary of all statistics.

        Returns:
            Dictionary containing summarized statistics and full history
        """
        summary = {
            # General statistics
            "total_steps": self.step_count,
            # Operations history
            "operations_history": {op_name: op_list for op_name, op_list in self.operations.items()},
            # Current values
            "current_object_transfers": self.obj_transfers.cpu().numpy().tolist(),
            "current_source_tote_ejections": self.source_tote_ejections.cpu().numpy().tolist(),
            "current_dest_tote_ejections": self.dest_tote_ejections.cpu().numpy().tolist(),
            # Comprehensive ejection history
            "ejection_snapshots": self.ejection_snapshots,
            # Source tote ejection history
            "source_tote_ejections_history": [
                {
                    "step": entry["step"],
                    "values": entry["values"].cpu().numpy().tolist(),
                    "timestamp": entry["timestamp"],
                }
                for entry in self.source_tote_ejections_history
            ],
            # Destination tote ejection history
            "dest_tote_ejections_history": [
                {
                    "step": entry["step"],
                    "values": entry["values"].cpu().numpy().tolist(),
                    "timestamp": entry["timestamp"],
                }
                for entry in self.dest_tote_ejections_history
            ],
            # Object transfer history
            "obj_transfers_history": [
                {
                    "step": entry["step"],
                    "values": entry["values"].cpu().numpy().tolist(),
                    "timestamp": entry["timestamp"],
                }
                for entry in self.obj_transfers_history
            ],
            # GCU history
            "gcu_history": [
                {
                    "step": entry["step"],
                    "values": entry["values"].cpu().numpy().tolist(),
                    "env_ids": entry["env_ids"].cpu().numpy().tolist(),
                    "timestamp": entry["timestamp"],
                }
                for entry in self.gcu_history
            ],
        }

        return summary

    def get_ejection_summary(self) -> dict[str, Any]:
        """
        Get a summary of the most recent ejection statistics.

        Returns:
            Dictionary containing the most recent ejection data
        """
        summary = {}
        # Check if there are any snapshots
        for env_id in range(self.num_envs):
            if self.ejection_snapshots[env_id] and len(self.ejection_snapshots[env_id]) > 0:
                # Return a summary of the last snapshot for this environment
                tote_ids = self.ejection_snapshots[env_id][-1][0]
                summary_env = self.ejection_snapshots[env_id][-1][1]
                last_gcu = (
                    summary_env["gcu_history"][-1]["values"][env_id][tote_ids]
                    if "gcu_history" in summary_env and len(summary_env["gcu_history"]) > 0
                    else None
                )
                last_obj_transfers = (
                    summary_env["obj_transfers_history"][-1]["values"][env_id]
                    if "obj_transfers_history" in summary_env and len(summary_env["obj_transfers_history"]) > 0
                    else None
                )
                last_source_ejection = (
                    summary_env["source_tote_ejections_history"][-1]["values"][env_id]
                    if "source_tote_ejections_history" in summary_env
                    and len(summary_env["source_tote_ejections_history"]) > 0
                    else None
                )
                summary[env_id] = {}
                summary[env_id]["last_gcu"] = last_gcu
                summary[env_id]["last_source_ejection"] = last_source_ejection
                summary[env_id]["last_obj_transfers"] = last_obj_transfers

        return summary

    def save_to_file(self, filepath: str):
        """
        Save statistics to a file.

        Args:
            filepath: Path to save the statistics
        """
        import json

        # Convert data to JSON-serializable format
        summary = self.get_summary()

        # Save to file
        with open(filepath, "w") as f:
            json.dump(summary, f, indent=2)

        logger.info(f"Statistics saved to {filepath}")
