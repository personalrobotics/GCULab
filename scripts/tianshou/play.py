# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to train RL agent with RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher
import arguments
from ts_train import build_net


# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--max_iterations", type=int, default=None, help="RL Policy training iterations.")
parser.add_argument(
    "--distributed", action="store_true", default=False, help="Run training with multiple GPUs or nodes."
)

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import os
from datetime import datetime

import gymnasium as gym
import isaaclab_tasks  # noqa: F401
import torch
import tote_consolidation.tasks  # noqa: F401
from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_pickle, dump_yaml
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config
from tianshou.utils.net.common import ActorCritic
from tianshou.utils import TensorboardLogger, LazyLogger
from torch.optim.lr_scheduler import LambdaLR, ExponentialLR
from torch.utils.tensorboard import SummaryWriter

from tools import *
from masked_ppo import MaskedPPOPolicy
from masked_a2c import MaskedA2CPolicy
from isaacpackcollector import IsaacPackCollector
from vecenv_wrapper import TianShouVecEnvWrapper
import tianshou as ts


torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False

from omegaconf import OmegaConf

def make_envs(args, obs):
    test_envs = ts.env.IsaacSubprocVectorEnv(
        [lambda: gym.make(args.env.id, 
                          container_size=args.env.container_size,
                          enable_rotation=args.env.rot,
                          data_type=args.env.box_type,
                          item_set=args.env.box_size_set, 
                          reward_type=args.train.reward_type,
                          action_scheme=args.env.scheme,
                          k_placement=args.env.k_placement) 
                          for _ in range(args.train.num_processes)]
    )

    test_envs.seed(args.seed, next_box=obs['policy'][:, -3:].detach().cpu().numpy()[:, [2, 1, 0]], heightmap=depth_to_heightmap(obs['sensor'].squeeze().detach().cpu().numpy()))

    return test_envs, None

@hydra_task_config(args_cli.task, "tianshou_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: dict):
    """Train with RSL-RL agent."""

    args = OmegaConf.create(agent_cfg)

    # set the environment seed
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs

    args.train.num_processes = env_cfg.scene.num_envs
    args.train.step_per_collect = args.train.num_processes * args.train.num_steps  


    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = args.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "tianshou", args.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Logging experiment in directory: {log_root_path}")
    # specify directory for logging runs: {time-stamp}_{run_name}
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    # The Ray Tune workflow extracts experiment name using the logging line below, hence, do not change it (see PR #2346, comment-2819298849)
    print(f"Exact experiment name requested from command line: {log_dir}")
    if args.experiment_name:
        log_dir += f"_{args.experiment_name}"
    log_dir = os.path.join(log_root_path, log_dir)

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    env = TianShouVecEnvWrapper(env, make_envs_args=args)
    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "train"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    date = time.strftime(r'%Y.%m.%d-%H-%M-%S', time.localtime(time.time()))
    time_str = args.env.id + "_" + \
        str(args.env.container_size[0]) + "-" + str(args.env.container_size[1]) + "-" + str(args.env.container_size[2]) + "_" + \
        args.env.scheme + "_" + str(args.env.k_placement) + "_" +\
        args.env.box_type + "_" + \
        args.train.algo  + '_' \
        'seed' + str(args.seed) + "_" + \
        args.opt.optimizer + "_" \
        + date

    device = torch.device("cuda", 0)

    obs, extras = env.get_observations()

    # Calculate box size parameters
    box_small = int(max(args.env.container_size) / 10)
    box_big = int(max(args.env.container_size) / 2)
    box_range = (box_small, box_small, box_small, box_big, box_big, box_big)

    if args.env.get("step") is not None:
        step = args.env.step
    else:
        step = box_small

    box_size_set = []
    for i in range(box_range[0], box_range[3] + 1, step):
        for j in range(box_range[1], box_range[4] + 1, step):
            for k in range(box_range[2], box_range[5] + 1, step):
                box_size_set.append((i, j, k))
    
    args.env.box_small = box_small
    args.env.box_big = box_big
    args.env.box_size_set = box_size_set

    # environments
    test_envs = make_envs(args, extras['observations'])  # make envs and set random seed

    # network
    actor, critic = build_net(args, device)
    actor_critic = ActorCritic(actor, critic)

    # Ensure numeric parameters are properly typed
    args.opt.lr = float(args.opt.lr)
    args.opt.eps = float(args.opt.eps)
    if hasattr(args.opt, 'alpha'):
        args.opt.alpha = float(args.opt.alpha)
    args.loss.entropy = float(args.loss.entropy)
    args.loss.value = float(args.loss.value)
    args.train.gae_lambda = float(args.train.gae_lambda)
    args.train.gamma = float(args.train.gamma)
    args.train.clip_param = float(args.train.clip_param)

    if args.opt.optimizer == 'Adam':
        optim = torch.optim.Adam(actor_critic.parameters(), lr=args.opt.lr, eps=args.opt.eps)
    elif args.opt.optimizer == 'RMSprop':
        optim = torch.optim.RMSprop(
            actor_critic.parameters(),
            lr=args.opt.lr,
            eps=args.opt.eps,
            alpha=args.opt.alpha,
        )
    else:
        raise NotImplementedError

    args.train.step_per_collect = args.train.num_processes * args.train.num_steps

    lr_scheduler = None
    if args.opt.lr_decay:
        # decay learning rate to 0 linearly
        max_update_num = np.ceil(args.train.step_per_epoch / args.train.step_per_collect) * args.train.epoch
        lr_scheduler = LambdaLR(optim, lr_lambda=lambda epoch: 1 - epoch / max_update_num)


    # RL agent 
    dist = CategoricalMasked
    if args.train.algo == 'PPO':
        policy = MaskedPPOPolicy(
            actor=actor,
            critic=critic,
            optim=optim,
            dist_fn=dist,
            discount_factor=args.train.gamma,
            eps_clip=args.train.clip_param,
            advantage_normalization=False,
            vf_coef=args.loss.value,
            ent_coef=args.loss.entropy,
            gae_lambda=args.train.gae_lambda,
            lr_scheduler=lr_scheduler
        )
    elif args.algo == 'A2C':    
        policy = MaskedA2CPolicy(
            actor,
            critic,
            optim,
            dist,
            discount_factor=args.train.gamma,
            vf_coef=args.loss.value,
            ent_coef=args.loss.entropy,
            gae_lambda=args.train.gae_lambda,
            lr_scheduler=lr_scheduler
        )
    else:
        raise NotImplementedError

    
    policy.eval()
    try:
        policy.load_state_dict(torch.load(args.ckp, map_location=device))
        # print(policy)
    except FileNotFoundError:
        print("No model found")
        exit()

    log_path = './logs/' + time_str
    
    is_debug = True if sys.gettrace() else False
    if not is_debug:
        writer = SummaryWriter(log_path)
        logger = TensorboardLogger(
            writer=writer,
            train_interval=args.log_interval,
            update_interval=args.log_interval
        )
    else:
        logger = LazyLogger()

    test_collector = IsaacPackCollector(policy, test_envs, env)

    result = test_collector.collect(n_episode=args.test_episode, render=args.render)
    for i in range(args.test_episode):
        print(f"episode {i+1}\t => \tratio: {result['ratios'][i]:.4f} \t| total: {result['nums'][i]}")
    print('All cases have been done!')
    print('----------------------------------------------')
    print('average space utilization: %.4f'%(result['ratio']))
    print('average put item number: %.4f'%(result['num']))
    print("standard variance: %.4f"%(result['ratio_std']))

if __name__ == "__main__":
    # run the main function
    registration_envs()
    args = arguments.get_args()
    args.train.algo = args.train.algo.upper()
    main()
    # close sim app
    simulation_app.close()
