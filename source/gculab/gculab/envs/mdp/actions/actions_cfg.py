# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import MISSING

from isaaclab.managers.action_manager import ActionTerm, ActionTermCfg
from isaaclab.utils import configclass

from . import packing_actions


@configclass
class PackingActionCfg(ActionTermCfg):
    """Configuration for the packing action term.

    See :class:`PackingAction` for more details.
    """

    class_type: type[ActionTerm] = packing_actions.PackingAction
