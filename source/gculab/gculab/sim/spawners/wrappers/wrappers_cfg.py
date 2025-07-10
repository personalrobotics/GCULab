# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import MISSING

from isaaclab.sim.spawners.from_files import UsdFileCfg
from isaaclab.sim.spawners.spawner_cfg import DeformableObjectSpawnerCfg, RigidObjectSpawnerCfg, SpawnerCfg
from isaaclab.utils import configclass
import torch
from . import wrappers
from isaaclab.sim import MultiUsdFileCfg

@configclass
class MultiUsdFromDistFileCfg(UsdFileCfg):
    """Configuration parameters for loading multiple USD files.

    Specifying values for any properties at the configuration level is applied to all the assets
    imported from their USD files.

    .. tip::
        It is recommended that all the USD based assets follow a similar prim-hierarchy.

    """

    func = wrappers.spawn_multi_usd_from_dist_file

    usd_path: str | list[str] = MISSING
    """Path or a list of paths to the USD files to spawn asset from."""

    random_choice: bool = True
    """Whether to randomly select an asset configuration. Default is True.

    If False, the asset configurations are spawned in the order they are provided in the list.
    If True, a random asset configuration is selected for each spawn.
    """

    distribution: torch.Tensor | None = None
    """Distribution tensor to sample from when spawning assets.
    The distribution tensor should have the same length as the number of USD files,
    and should contain normalized probabilities for selecting each USD file.
    """