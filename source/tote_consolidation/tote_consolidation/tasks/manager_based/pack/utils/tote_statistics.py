# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import json
import os
import pickle
import queue
import threading
import time
from collections import defaultdict, deque
from typing import Any, Optional

import torch
from packing3d import Container


class ToteStatistics:
    """
    Class for tracking and logging tote operation statistics.

    This tracks metrics like GCU over time, object transfers,
    and ejection counts for source and destination totes.
    """

    def __init__(self, num_envs: int, num_totes: int, device: torch.device, save_path: str | None = None, disable_logging: bool = False):
        """
        Initialize the tote statistics tracker.

        Args:
            num_envs: Number of environments
            num_totes: Number of totes per environment
            device: Device for torch tensors
            save_path: Optional path to incrementally save statistics
            disable_logging: If True, disable all file logging for maximum performance
        """
        self.num_envs = num_envs
        self.num_totes = num_totes
        self.device = device
        self.save_path = save_path
        self.disable_logging = disable_logging

        # Initialize file if save_path is provided and logging is not disabled
        if save_path and not disable_logging:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            with open(save_path, "w") as f:
                json.dump({"environments": {str(i): {} for i in range(num_envs)}, "summary": {}}, f, indent=4)
            self.pkl_dir = os.path.join(os.path.dirname(save_path), "containers")
            os.makedirs(self.pkl_dir, exist_ok=True)
        else:
            self.pkl_dir = None

        # Initialize statistics trackers
        self.step_count = 0

        # Keep only recent GCU values instead of full history
        self.recent_gcu_values = None
        self.recent_gcu_env_ids = None

        # Track operation counts rather than full history
        self.operation_counts = defaultdict(int)

        # Object transfer counts per environment
        self.obj_transfers = torch.zeros(num_envs, device=device)

        # Ejection counts per environment
        self.source_tote_ejections = torch.zeros(num_envs, device=device)
        self.dest_tote_ejections = torch.zeros(num_envs, device=device)

        # Inbound and outbound GCU values for each tote
        self.inbound_gcus = [[deque() for _ in range(num_totes)] for _ in range(num_envs)]
        self.outbound_gcus = [[deque() for _ in range(num_totes)] for _ in range(num_envs)]

        self.containers = [None for _ in range(num_envs)]

        # Track only the most recent ejection snapshot for each environment
        self.recent_ejection_data = [None for _ in range(self.num_envs)]

        # Timestamp when logging started
        self.start_time = time.time()

        # Batched logging infrastructure
        self.log_queue = queue.Queue()
        self.log_thread = None
        self.shutdown_event = threading.Event()

        if save_path and not disable_logging:
            self._start_logging_thread()

    def _start_logging_thread(self):
        """Start the background logging thread."""
        self.log_thread = threading.Thread(target=self._logging_worker, daemon=True)
        self.log_thread.start()

    def _logging_worker(self):
        """Background worker thread that handles file I/O operations."""
        # In-memory cache of file data to avoid repeated reads
        file_cache = {"environments": {str(i): {} for i in range(self.num_envs)}, "summary": {}}
        cache_dirty = False

        while not self.shutdown_event.is_set():
            try:
                # Get log entry with timeout to allow checking shutdown event
                log_entry = self.log_queue.get(timeout=1.0)

                if log_entry is None:  # Shutdown signal
                    break

                # Check if this is a pickle file entry or JSON log entry
                if isinstance(log_entry, tuple) and len(log_entry) == 3 and log_entry[0] == "pickle":
                    # Handle pickle file write
                    _, pkl_path, container = log_entry
                    try:
                        with open(pkl_path, "wb") as f:
                            pickle.dump(container, f)
                    except Exception as e:
                        print(f"Error writing pickle file {pkl_path}: {e}")
                else:
                    # Handle JSON log entry
                    env_id, data_type, data = log_entry

                    # Update cache
                    env_key = str(env_id)
                    if env_key not in file_cache["environments"]:
                        file_cache["environments"][env_key] = {}
                    if data_type not in file_cache["environments"][env_key]:
                        file_cache["environments"][env_key][data_type] = []

                    file_cache["environments"][env_key][data_type].append(data)
                    cache_dirty = True

                # Periodically flush to disk (every 10 entries or when queue is empty)
                if self.log_queue.qsize() == 0 or cache_dirty:
                    try:
                        with open(self.save_path, "w") as f:
                            json.dump(file_cache, f, indent=4)
                        cache_dirty = False
                    except Exception as e:
                        print(f"Error writing to log file: {e}")

            except queue.Empty:
                # Flush any remaining data when queue is empty
                if cache_dirty:
                    try:
                        with open(self.save_path, "w") as f:
                            json.dump(file_cache, f, indent=4)
                        cache_dirty = False
                    except Exception as e:
                        print(f"Error writing to log file: {e}")

        # Final flush on shutdown
        if cache_dirty:
            try:
                with open(self.save_path, "w") as f:
                    json.dump(file_cache, f, indent=4)
            except Exception as e:
                print(f"Error writing to log file during shutdown: {e}")

    def shutdown(self):
        """Properly shutdown the logging thread and flush all data."""
        if self.log_thread and self.log_thread.is_alive():
            self.shutdown_event.set()
            try:
                self.log_queue.put(None, timeout=1.0)  # Signal shutdown with timeout
            except queue.Full:
                print("Warning: Log queue full, could not signal shutdown to logging thread")
            self.log_thread.join(timeout=5.0)
            if self.log_thread.is_alive():
                print("Warning: Logging thread did not shutdown cleanly")

    def __del__(self):
        """Cleanup logging thread on destruction."""
        self.shutdown()

    def _append_to_file(self, env_id, data_type, data):
        """
        Queue data for background logging instead of synchronous file I/O.

        Args:
            env_id: Environment ID
            data_type: Type of data (e.g., 'gcu', 'transfers')
            data: Data to append
        """
        if not self.save_path or self.log_thread is None or self.disable_logging:
            return

        try:
            # Queue the log entry for background processing
            self.log_queue.put((env_id, data_type, data), block=False)
        except queue.Full:
            # If queue is full, skip this log entry to avoid blocking
            print(f"Warning: Log queue full, skipping log entry for env {env_id}")
        except Exception as e:
            print(f"Error queuing log entry: {e}")

    def log_gcu(self, gcu_values: torch.Tensor, env_ids: torch.Tensor = None):
        """
        Log GCU values for specified environments.

        Args:
            gcu_values: Tensor of GCU values [num_envs, num_totes]
            env_ids: Optional tensor of environment IDs to log for
        """
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)

        # Store only the most recent GCU values
        self.recent_gcu_values = gcu_values.detach().clone()
        self.recent_gcu_env_ids = env_ids.detach().clone()

        # Save data to file if path is provided
        if self.save_path and not self.disable_logging:
            current_time = time.time() - self.start_time
            for i, env_id in enumerate(env_ids.cpu().numpy()):
                self._append_to_file(
                    env_id,
                    "gcu_values",
                    {
                        "step": self.step_count,
                        "values": gcu_values[i].cpu().numpy().tolist(),
                        "timestamp": current_time,
                    },
                )

    def log_obj_transfers(self, env_ids: torch.Tensor, num_objects: int = 1):
        """
        Log object transfer operations.

        Args:
            env_ids: Tensor of environment IDs where transfers occurred
            num_objects: Number of objects placed/transferred
        """
        self.obj_transfers[env_ids] += num_objects
        self.operation_counts["object_transfers"] += len(env_ids)

        # Save data to file if path is provided
        if self.save_path and not self.disable_logging:
            current_time = time.time() - self.start_time
            for env_id in env_ids.cpu().numpy():
                self._append_to_file(
                    env_id,
                    "transfers",
                    {
                        "step": self.step_count,
                        "count": num_objects,
                        "total": self.obj_transfers[env_id].item(),
                        "timestamp": current_time,
                    },
                )

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

            # Save to file
            if self.save_path and not self.disable_logging:
                self._append_to_file(
                    env.item(),
                    "tote_gcu_changes",
                    {
                        "step": self.step_count,
                        "tote_id": tote.item(),
                        "inbound_gcu": inbound_gcus[env, tote].item(),
                        "outbound_gcu": outbound_gcus[env, tote].item(),
                        "timestamp": time.time() - self.start_time,
                    },
                )

    def log_source_tote_ejection(self, env_ids: torch.Tensor):
        """
        Log source tote ejection events.

        Args:
            env_ids: Tensor of environment IDs where ejections occurred
        """
        if self.step_count == 0:
            return  # Skip logging on the first step
        self.source_tote_ejections[env_ids] += 1
        self.operation_counts["source_tote_ejection"] += len(env_ids)

        # Save to file
        if self.save_path and not self.disable_logging:
            current_time = time.time() - self.start_time
            for env_id in env_ids.cpu().numpy():
                self._append_to_file(
                    env_id,
                    "source_ejections",
                    {
                        "step": self.step_count,
                        "count": 1,
                        "total": self.source_tote_ejections[env_id].item(),
                        "timestamp": current_time,
                    },
                )

    def log_container(self, env_idx: int, container: Container):
        self.containers[env_idx] = container

        if self.save_path and self.pkl_dir and not self.disable_logging:
            current_time = time.time() - self.start_time

            # Create per-env subdir
            env_dir = os.path.join(self.pkl_dir, f"env_{env_idx}")
            os.makedirs(env_dir, exist_ok=True)

            # Save the container as step.pkl (e.g., 42.pkl)
            pkl_filename = f"{self.step_count}.pkl"
            pkl_path = os.path.join(env_dir, pkl_filename)

            # Queue the pickle file write for background processing
            try:
                self.log_queue.put(("pickle", pkl_path, container), block=False)
            except queue.Full:
                print(f"Warning: Log queue full, skipping container save for env {env_idx}")

            # Log pointer to JSON
            self._append_to_file(
                env_idx,
                "container",
                {
                    "step": self.step_count,
                    "pickle_file": f"env_{env_idx}/{pkl_filename}",
                    "timestamp": current_time,
                },
            )

    def log_dest_tote_ejection(self, tote_ids: torch.Tensor, env_ids: torch.Tensor):
        """
        Log destination tote ejection events.

        Args:
            tote_ids: Tensor of tote IDs where ejections occurred
            env_ids: Tensor of environment IDs where ejections occurred
        """
        self.dest_tote_ejections[env_ids] += 1
        self.operation_counts["dest_tote_ejection"] += len(env_ids)

        # Save snapshots for each environment
        for idx, env_id in enumerate(env_ids):
            env_id_item = env_id.item()
            tote_id_item = tote_ids[idx].item()

            # Create a minimal snapshot with just the essential data
            snapshot_data = {
                "step": self.step_count,
                "tote_id": tote_id_item,
                "obj_transfers": self.obj_transfers[env_id].item(),
                "source_ejections": self.source_tote_ejections[env_id].item(),
                "gcu": (
                    self.recent_gcu_values[env_id, tote_id_item].item() if self.recent_gcu_values is not None else None
                ),
                "timestamp": time.time() - self.start_time,
            }

            # Store only the most recent snapshot
            self.recent_ejection_data[env_id_item] = snapshot_data

            # Save to file
            if self.save_path and not self.disable_logging:
                self._append_to_file(env_id_item, "dest_ejections", snapshot_data)

            # Reset counts for this environment
            self.source_tote_ejections[env_id] = 0
            self.dest_tote_ejections[env_id] = 0
            self.obj_transfers[env_id] = 0

    def increment_step(self):
        """Increment the step counter."""
        self.step_count += 1

    def get_summary(self, env_ids: torch.Tensor | int | None = None) -> dict[str, Any]:
        """
        Get a summary of statistics for specified environment IDs.

        Args:
            env_ids: Environment IDs to include in the summary. Can be a tensor, single integer, or None (for all environments)

        Returns:
            Dictionary containing summarized statistics for specified environments
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

        # Create a summary with current values
        summary = {
            "total_steps": self.step_count,
            "operation_counts": dict(self.operation_counts),
            "current_object_transfers": self.obj_transfers[env_ids].cpu().numpy().tolist(),
            "current_source_tote_ejections": self.source_tote_ejections[env_ids].cpu().numpy().tolist(),
            "current_dest_tote_ejections": self.dest_tote_ejections[env_ids].cpu().numpy().tolist(),
        }

        # Add recent GCU values if available
        if self.recent_gcu_values is not None:
            summary["recent_gcu_values"] = self.recent_gcu_values[env_ids].cpu().numpy().tolist()

        # Add recent ejection data
        summary["recent_ejections"] = [
            self.recent_ejection_data[i]
            for i in env_id_list
            if i < len(self.recent_ejection_data) and self.recent_ejection_data[i] is not None
        ]

        return summary

    def get_ejection_summary(self) -> dict[str, Any]:
        """
        Get a summary of the most recent ejection statistics.

        Returns:
            Dictionary containing the most recent ejection data
        """
        summary = {}
        for env_id in range(self.num_envs):
            if self.recent_ejection_data[env_id] is not None:
                summary[env_id] = {
                    "last_gcu": self.recent_ejection_data[env_id].get("gcu"),
                    "last_source_ejection": self.recent_ejection_data[env_id].get("source_ejections"),
                    "last_obj_transfers": self.recent_ejection_data[env_id].get("obj_transfers"),
                }
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
            if self.recent_ejection_data[env_id] is not None:
                # Read from file if available, otherwise use in-memory data
                if self.save_path and not self.disable_logging:
                    try:
                        with open(self.save_path) as f:
                            file_data = json.load(f)
                            env_data = file_data["environments"].get(str(env_id), {})
                            dest_ejections = env_data.get("dest_ejections", [])

                            if dest_ejections:
                                summary[env_id] = {
                                    "gcus": [
                                        entry.get("gcu") for entry in dest_ejections if entry.get("gcu") is not None
                                    ],
                                    "obj_transfers": [entry.get("obj_transfers", 0) for entry in dest_ejections],
                                    "source_ejections": [entry.get("source_ejections", 0) for entry in dest_ejections],
                                }

                                # Update means
                                for entry in dest_ejections:
                                    gcu = entry.get("gcu")
                                    if gcu is not None:
                                        mean_gcus += gcu
                                    mean_obj_transfers += entry.get("obj_transfers", 0)
                                    mean_source_ejections += entry.get("source_ejections", 0)
                                    data_count += 1
                    except Exception as e:
                        print(f"Error reading from file: {e}")
                        # Fallback to in-memory data
                        recent_data = self.recent_ejection_data[env_id]
                        summary[env_id] = {
                            "gcus": [recent_data.get("gcu")] if recent_data.get("gcu") is not None else [],
                            "obj_transfers": [recent_data.get("obj_transfers", 0)],
                            "source_ejections": [recent_data.get("source_ejections", 0)],
                        }

                        gcu = recent_data.get("gcu")
                        if gcu is not None:
                            mean_gcus += gcu
                        mean_obj_transfers += recent_data.get("obj_transfers", 0)
                        mean_source_ejections += recent_data.get("source_ejections", 0)
                        data_count += 1
                else:
                    # Use in-memory data only
                    recent_data = self.recent_ejection_data[env_id]
                    summary[env_id] = {
                        "gcus": [recent_data.get("gcu")] if recent_data.get("gcu") is not None else [],
                        "obj_transfers": [recent_data.get("obj_transfers", 0)],
                        "source_ejections": [recent_data.get("source_ejections", 0)],
                    }

                    gcu = recent_data.get("gcu")
                    if gcu is not None:
                        mean_gcus += gcu
                    mean_obj_transfers += recent_data.get("obj_transfers", 0)
                    mean_source_ejections += recent_data.get("source_ejections", 0)
                    data_count += 1

        # Calculate means
        summary["mean_gcus"] = mean_gcus / data_count if data_count > 0 else 0
        summary["mean_obj_transfers"] = mean_obj_transfers / data_count if data_count > 0 else 0
        summary["mean_source_ejections"] = mean_source_ejections / data_count if data_count > 0 else 0

        return summary

    def flush_logs(self):
        """Flush all queued log entries to disk."""
        if self.log_thread and self.log_thread.is_alive():
            # Wait for queue to be processed
            while not self.log_queue.empty():
                time.sleep(0.01)  # Small delay to allow background thread to process
            time.sleep(0.1)  # Additional time to ensure final flush

    def save_to_file(self, filepath: str = None):
        """
        Save statistics to a file.

        Args:
            filepath: Path to save the statistics (if None, uses self.save_path with a different filename)
        """
        # Flush any pending logs first
        self.flush_logs()

        # If no filepath provided, use the one set during initialization
        if filepath is None:
            if self.save_path is None:
                raise ValueError("No filepath provided for saving statistics")

            # Generate a new filename in the same directory with a different name
            base_dir = os.path.dirname(self.save_path)
            base_name = os.path.splitext(os.path.basename(self.save_path))[0]
            new_filename = f"{base_name}_summary.json"
            filepath = os.path.join(base_dir, new_filename)

        # Get summary data
        summary = self.get_full_ejection_summary()

        # If we've been incrementally saving, just update the summary section
        if filepath == self.save_path and os.path.exists(filepath):
            try:
                with open(filepath, "r+") as f:
                    file_data = json.load(f)
                    file_data["summary"] = summary
                    f.seek(0)
                    f.truncate()
                    json.dump(file_data, f, indent=4)
                return
            except Exception as e:
                print(f"Error updating summary in file: {e}")
                # Fall back to writing the whole file

        # Write the whole file if needed
        with open(filepath, "w") as f:
            json.dump(summary, f, indent=4)
