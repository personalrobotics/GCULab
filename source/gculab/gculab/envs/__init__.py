# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Sub-package for environment definitions.

Environments define the interface between the agent and the simulation.
In the simplest case, the environment provides the agent with the current
observations and executes the actions provided by the agent. However, the
environment can also provide additional information such as the current
reward, done flag, and information about the current episode.

"""

from . import mdp  # noqa: F401, F403
from .manager_based_rl_gcu_env import ManagerBasedRLGCUEnv
