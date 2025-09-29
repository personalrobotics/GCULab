# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import isaaclab.sim as sim_utils
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import TiledCameraCfg
from isaaclab.utils import configclass

from . import mdp
from .pack_env_cfg import PackEnvCfg, PackSceneCfg

##
# Scene definition
##


@configclass
class PackRGBCameraSceneCfg(PackSceneCfg):
    """Configuration for the scene with a robotic arm."""

    tiled_camera: TiledCameraCfg = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/Camera",
        offset=TiledCameraCfg.OffsetCfg(pos=(0.45, -0.645, 20.0), rot=(0.0, 0.0, 0.0, -1.0), convention="opengl"),
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=810.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 21.0)
        ),
        width=52,
        height=37,
    )


@configclass
class PackDepthCameraSceneCfg(PackSceneCfg):
    """Configuration for the scene with a robotic arm."""

    tiled_camera: TiledCameraCfg = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/Camera",
        offset=TiledCameraCfg.OffsetCfg(pos=(0.45, -0.645, 20.0), rot=(0.0, 0.0, 0.0, -1.0), convention="opengl"),
        data_types=["distance_to_camera"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=810.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 21.0)
        ),
        width=52,
        height=37,
    )


@configclass
class RGBObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class RGBCameraPolicyCfg(ObsGroup):
        """Observations for policy group with RGB images."""

        image = ObsTerm(func=mdp.image, params={"sensor_cfg": SceneEntityCfg("tiled_camera"), "data_type": "rgb"})

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: ObsGroup = RGBCameraPolicyCfg()


@configclass
class DepthObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)
        # actions = ObsTerm(func=mdp.last_action)
        # obs_dims = ObsTerm(func=mdp.obs_dims)
        obs_latents = ObsTerm(func=mdp.obs_latents)

        # def __post_init__(self):
        #     self.enable_corruption = True
        #     self.concatenate_terms = True

    class SensorCfg(ObsGroup):
        """Observations for sensor group."""

        image = ObsTerm(
            func=mdp.image, params={"sensor_cfg": SceneEntityCfg("tiled_camera"), "data_type": "distance_to_camera"}
        )

    # observation groups
    policy: PolicyCfg = PolicyCfg()
    sensor: SensorCfg = SensorCfg()


@configclass
class ResNet18ObservationCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PackResNet18RGBCameraEnvCfg(ObsGroup):
        """Observations for policy group with features extracted from RGB images with a frozen ResNet18."""

        image = ObsTerm(
            func=mdp.image_features,
            params={"sensor_cfg": SceneEntityCfg("tiled_camera"), "data_type": "rgb", "model_name": "resnet18"},
        )

    @configclass
    class PackResNet18DepthCameraEnvCfg(ObsGroup):
        """Observations for policy group with features extracted from depth images with a frozen ResNet18."""

        image = ObsTerm(
            func=mdp.image_features,
            params={
                "sensor_cfg": SceneEntityCfg("tiled_camera"),
                "data_type": "distance_to_camera",
                "model_name": "resnet18",
            },
        )

    policy: ObsGroup = PackResNet18DepthCameraEnvCfg()


@configclass
class PackRGBCameraEnvCfg(PackEnvCfg):
    """Configuration for the packing environment with RGB camera."""

    scene: PackRGBCameraSceneCfg = PackRGBCameraSceneCfg(num_envs=512, env_spacing=2.5, replicate_physics=False)
    observations: RGBObservationsCfg = RGBObservationsCfg()

    def __post_init__(self):
        super().__post_init__()
        # remove ground as it obstructs the camera
        self.scene.ground = None


@configclass
class PackDepthCameraEnvCfg(PackEnvCfg):
    """Configuration for the packing environment with depth camera."""

    scene: PackDepthCameraSceneCfg = PackDepthCameraSceneCfg(num_envs=512, env_spacing=2.5, replicate_physics=False)
    observations: DepthObservationsCfg = DepthObservationsCfg()

    def __post_init__(self):
        super().__post_init__()
        # remove ground as it obstructs the camera
        # self.scene.ground = None


@configclass
class PackResNet18DepthCameraEnvCfg(PackDepthCameraEnvCfg):
    """Configuration for the cartpole environment with ResNet18 features as observations."""

    observations: ResNet18ObservationCfg = ResNet18ObservationCfg()
