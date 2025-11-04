# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
from __future__ import annotations

import torch
from collections.abc import Sequence

from isaaclab.managers.recorder_manager import RecorderTerm

class PreStepFlatSensorObservationsRecorder(RecorderTerm):
    """Recorder term that records the sensor group observations in each step."""

    def record_pre_step(self):
        return "obs_sensor", self._env.obs_buf["sensor"]
