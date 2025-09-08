"""Env package."""

from tianshou.env.gym_wrappers import (
    ContinuousToDiscrete,
    MultiDiscreteToDiscrete,
    TruncatedAsTerminated,
)
from tianshou.env.venv_wrappers import VectorEnvNormObs, VectorEnvWrapper
from tianshou.env.venvs import (
    BaseVectorEnv,
    DummyVectorEnv,
    RayVectorEnv,
    ShmemVectorEnv,
    SubprocVectorEnv,
)
from tianshou.env.isaac_venv import IsaacSubprocVectorEnv

try:
    from tianshou.env.pettingzoo_env import PettingZooEnv
except ImportError:
    pass

__all__ = [
    "BaseVectorEnv",
    "DummyVectorEnv",
    "SubprocVectorEnv",
    "IsaacSubprocVectorEnv",
    "ShmemVectorEnv",
    "RayVectorEnv",
    "VectorEnvWrapper",
    "VectorEnvNormObs",
    "PettingZooEnv",
    "ContinuousToDiscrete",
    "MultiDiscreteToDiscrete",
    "TruncatedAsTerminated",
]
