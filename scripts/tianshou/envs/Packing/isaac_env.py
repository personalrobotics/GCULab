from typing import Optional

from .container import Container
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from .cutCreator import CuttingBoxCreator
# from .mdCreator import MDlayerBoxCreator
from .binCreator import RandomBoxCreator, LoadBoxCreator, BoxCreator
from .env import PackingEnv

from render import VTKRender


class IsaacPackingEnv(PackingEnv):
    def __init__(
        self,
        container_size=(10, 10, 10),
        item_set=None, 
        data_name=None, 
        load_test_data=False,
        enable_rotation=False,
        data_type="random",
        reward_type=None,
        action_scheme="heightmap",
        k_placement=100,
        is_render=False,
        is_hold_on=False,
        **kwags
    ) -> None:
        self.bin_size = container_size
        self.area = int(self.bin_size[0] * self.bin_size[1])
        # packing state
        self.container = Container(*self.bin_size, rotation=enable_rotation)
        self.can_rotate = enable_rotation
        self.reward_type = reward_type
        self.action_scheme = action_scheme
        self.k_placement = k_placement
        if action_scheme == "EMS":
            self.candidates = np.zeros((self.k_placement, 6), dtype=np.int32)  # (x1, y1, z1, x2, y2, H)
        else:
            self.candidates = np.zeros((self.k_placement, 3), dtype=np.int32)  # (x, y, z)

        # TODO (kaikwan): Removed self.box_creator, next_box to be pulled from Isaac not boxcreator
        # The following parameteres need to be edited to work with 

        # for rendering
        if is_render:
            self.renderer = VTKRender(container_size, auto_render=not is_hold_on)
        self.render_box = None
        self._set_space()

    @property
    def cur_observation(self):
        """
            get current observation and action mask
        """
        hmap = self.container.heightmap
        size = list(self.next_box)
        placements, mask = self.get_possible_position(size)
        self.candidates = np.zeros_like(self.candidates)
        if len(placements) != 0:
            # print("candidates:")
            # for c in placements:
            #     print(c)
            self.candidates[0:len(placements)] = placements
        size.extend([size[1], size[0], size[2]])
        obs = np.concatenate((hmap.reshape(-1), np.array(size).reshape(-1), self.candidates.reshape(-1)))
        mask = mask.reshape(-1)

        return {
            "obs": obs, 
            "mask": mask
        }

    def step(self, action, done, next_box, heightmap):
        """

        :param action: action index
        :return: cur_observation
                 reward
                 done, Whether to end boxing (i.e., the current box cannot fit in the bin)
                 info
        """
        self.next_box = np.round(np.array(next_box).squeeze() * 100).astype(int)
        self.container.heightmap = heightmap
        # print(self.next_box)
        pos, rot, size = self.idx2pos(action)
 
        # succeeded = self.container.place_box(self.next_box, pos, rot)
        
        if done:
            if self.reward_type == "terminal":  # Terminal reward
                reward = self.container.get_volume_ratio()
            else:  # Step-wise/Immediate reward
                reward = 0.0
            done = True
            
            self.render_box = [[0, 0, 0], [0, 0, 0]]
            info = {'counter': len(self.container.boxes), 'ratio': self.container.get_volume_ratio()}
            return self.cur_observation, reward, done, False, info

        box_ratio = self.get_box_ratio()

        # self.box_creator.drop_box()  # remove current box from the list
        # self.box_creator.generate_box_size()  # add a new box to the list

        if self.reward_type == "terminal":
            reward = 0.01
        else:
            reward = box_ratio
        done = False
        info = {'counter': len(self.container.boxes), 'ratio': self.container.get_volume_ratio()}
        return self.cur_observation, reward, done, False, info

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super(PackingEnv, self).reset(seed=seed)

        self.container = Container(*self.bin_size)
        # self.box_creator.generate_box_size()
        self.candidates = np.zeros_like(self.candidates)
        # Optionally handle options here if needed
        next_box = options.get('next_box') if options is not None else None
        if next_box is None:
            raise ValueError("next_box must be provided")
        heightmap = options.get('heightmap') if options is not None else None
        if heightmap is None:
            raise ValueError("heightmap must be provided")

        self.next_box = np.round(np.array(next_box).squeeze() * 100).astype(int)
        self.container.heightmap = heightmap
        return self.cur_observation, {}
