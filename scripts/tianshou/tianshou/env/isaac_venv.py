from typing import Any, Callable, List, Optional, Tuple, Union

import gymnasium as gym
import numpy as np
import torch
from tianshou.env.utils import ENV_TYPE

from tianshou.env.venvs import SubprocVectorEnv


try:
    import gym as old_gym

    has_old_gym = True
except ImportError:
    has_old_gym = False

GYM_RESERVED_KEYS = [
    "metadata", "reward_range", "spec", "action_space", "observation_space"
]



class IsaacSubprocVectorEnv(SubprocVectorEnv):
    def __init__(self, env_fns: List[Callable[[], ENV_TYPE]], **kwargs: Any) -> None:
        super().__init__(env_fns, **kwargs)

    def reset(
        self,
        id: Optional[Union[int, List[int], np.ndarray]] = None,
        next_box: torch.Tensor = None,
        heightmap: torch.Tensor = None,
        **kwargs: Any,
    ) -> Tuple[np.ndarray, Union[dict, List[dict]]]:
        """Reset the state of some envs and return initial observations.

        If id is None, reset the state of all the environments and return
        initial observations, otherwise reset the specific environments with
        the given id, either an int or a list.
        """
        self._assert_is_not_closed()
        id = self._wrap_id(id)
        if self.is_async:
            self._assert_id(id)

        if next_box is None:
            raise ValueError("next_box must be provided")
        if heightmap is None:
            raise ValueError("heightmap must be provided")

        # send(None) == reset() in worker
        for i in id:
            self.workers[i].send(None, **{'next_box': next_box[i], 'heightmap': heightmap[i]})
        ret_list = [self.workers[i].recv() for i in id]

        assert (
            isinstance(ret_list[0], (tuple, list)) and len(ret_list[0]) == 2
            and isinstance(ret_list[0][1], dict)
        )

        obs_list = [r[0] for r in ret_list]

        if isinstance(obs_list[0], tuple):  # type: ignore
            raise TypeError(
                "Tuple observation space is not supported. ",
                "Please change it to array or dict space",
            )
        try:
            obs = np.stack(obs_list)
        except ValueError:  # different len(obs)
            obs = np.array(obs_list, dtype=object)

        infos = [r[1] for r in ret_list]
        return obs, infos  # type: ignore

    def seed(
        self,
        seed: Optional[Union[int, List[int]]] = None,
        next_box: torch.Tensor = None,
        heightmap: torch.Tensor = None
    ) -> List[Optional[List[int]]]:
        """Set the seed for all environments.

        Accept ``None``, an int (which will extend ``i`` to
        ``[i, i + 1, i + 2, ...]``) or a list.

        :return: The list of seeds used in this env's random number generators.
            The first value in the list should be the "main" seed, or the value
            which a reproducer pass to "seed".
        """
        self._assert_is_not_closed()
        seed_list: Union[List[None], List[int]]
        if seed is None:
            seed_list = [seed] * self.env_num
        elif isinstance(seed, int):
            seed_list = [seed + i for i in range(self.env_num)]
        else:
            seed_list = seed
        if next_box is None:
            raise ValueError("next_box must be provided")
        if heightmap is None:
            raise ValueError("heightmap must be provided")

        return [w.seed(s, nb, hm) for w, s, nb, hm in zip(self.workers, seed_list, next_box, heightmap)]

    def map_action(
            self,
            action: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Map the action from the environment's action space to the IsaacGym action space.

        :param action: The action to be mapped.
        :return: The mapped position and rotation actions.
        """
        return [w.map_action(a) for w, a in zip(self.workers, action)]