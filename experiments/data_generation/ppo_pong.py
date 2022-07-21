import os
from typing import Optional
import argparse

import einops
import gym
import numpy as np
import stable_baselines3 as sb3
import torch
import torch.nn as nn
from einops.layers.torch import Rearrange
from gym_minigrid.wrappers import FullyObsWrapper
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.atari_wrappers import AtariWrapper, NoopResetEnv, EpisodicLifeEnv
from stable_baselines3.common.vec_env.base_vec_env import VecEnvWrapper
from torchvision.transforms import Grayscale, Resize
from wandb.integration.sb3 import WandbCallback

import matplotlib.pyplot as plt
import wandb


def conv_out(h_in, padding, kernel_size, stride):
    return int((h_in + 2. * padding - (kernel_size - 1.) - 1.) / stride + 1.)

def output_padding(h_in, conv_out, padding, kernel_size, stride):
    return h_in - (conv_out - 1) * stride + 2 * padding - (kernel_size - 1) - 1

def conv_out_shape(h_in, padding, kernel_size, stride):
    return tuple(conv_out(x, padding, kernel_size, stride) for x in h_in)

class ConvFeatureExtractor(BaseFeaturesExtractor):
    def __init__(
        self, 
        observation_space: gym.spaces.Box, 
        feature_dim: Optional[int] = 32,
        num_channels: Optional[int] = 32,
    ):

        super().__init__(observation_space, feature_dim)

        input_shape = observation_space.shape
        
        # input_shape = (observation_space.shape[-1], observation_space.shape[0], observation_space.shape[1])
        print(f"input_shape: {input_shape}")

        d = num_channels
        k = 3
        stride = 2
        conv1_shape = conv_out_shape(input_shape[1:], 0, k, stride)
        conv2_shape = conv_out_shape(conv1_shape, 0, k, stride)
        conv3_shape = conv_out_shape(conv2_shape, 0, k, stride)
        conv4_shape = conv_out_shape(conv3_shape, 0, k, stride)
        self.conv_shape = (d, *conv4_shape)
        print(f"{self.conv_shape = }")
        
        # self.net = nn.Sequential(
        #     Rearrange('b c h w -> b (c h w)'),
        #     nn.Linear(input_shape[0]*84*84, 256),
        #     nn.ReLU(),
        #     nn.Linear(256, 128),
        #     nn.ReLU(),
        #     nn.Linear(128, 64),
        #     nn.ReLU(),
        #     nn.Linear(64, feature_dim),
        # )
        
        self.net = nn.Sequential(
            nn.Conv2d(input_shape[0], 8*d, k, stride),
            nn.ReLU(),
            nn.Conv2d(8*d, 4*d, k, stride),
            nn.ReLU(),
            nn.Conv2d(4*d, 2*d, k, stride),
            nn.ReLU(),
            nn.Conv2d(2*d, d, k, stride),
            Rearrange('b ... -> b (...)'),
            nn.Linear(self.conv_shape[0] * self.conv_shape[1] * self.conv_shape[2], feature_dim),
        )
        
        print("\nFeature Extractor net:")
        print(self.net)
        
    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        obs = self.net(observations)
        return obs

class CropGrayWrapper(gym.ObservationWrapper):
    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.env = env
        self.trafo_list = [
            lambda x: x[:, 35:-15],
            lambda x: x/255,
            Resize((84,84)),
            Grayscale(),
            lambda x: x - 0.5,
            lambda x: x * 2,
        ]
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(1, 84, 84), dtype=np.float32)
        
    def trafos(self, x):
        old_x = x.clone()
        for i, trafo in enumerate(self.trafo_list):
            x = trafo(x)
        return x
        
    def observation(self, observation):
        observation = torch.from_numpy(einops.rearrange(observation, 'h w c -> c h w')).float()
        observation = self.trafos(observation).numpy() # extract channel

        return observation
    
def make_env(env_id, rank, seed=0):
        """
        Utility function for multiprocessed env.

        :param env_id: (str) the environment ID
        :param seed: (int) the inital seed for RNG
        :param rank: (int) index of the subprocess
        """
        def _init():
            env = gym.make(env_id)
            env.seed(seed + rank)
            return CropGrayWrapper(EpisodicLifeEnv(NoopResetEnv(env)))
        set_random_seed(seed)
        return _init

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_stack', type=int, default=4)
    parser.add_argument('--num_steps', type=int, default=10_000_000)
    parser.add_argument('--eval_freq', type=int, default=100_000)
    parser.add_argument('--num_envs', type=int, default=2)
    kwargs = vars(parser.parse_args())
    
    
    config = {
        "policy_type": "MlpPolicy",
        "total_timesteps": kwargs['num_steps'],
        "env_name": "Pong-v0",
        "policy_kwargs": dict(
            features_extractor_class=ConvFeatureExtractor,
        ),
        "batch_size": 256,
        "vf_coef": 0.5,
        "ent_coef": 0.01,
        "num_channels": 16,
        'feature_dim': 128,
        "network_dim": 512,
        'n_epochs':4,
        'learning_rate':0.0001,
    }

    config['policy_kwargs']['features_extractor_kwargs'] = dict(feature_dim=config['feature_dim'], num_channels=config['num_channels'])
    config['policy_kwargs']['net_arch'] = [dict(pi=[config['feature_dim'], config['network_dim']], vf=[config['feature_dim'], config['network_dim']])]

    channels_order = "first" # CropGrayWrapper
    
    venv= [make_env(config["env_name"], rank, seed=0) for rank in range(kwargs['num_envs'])]
    venv = DummyVecEnv(venv)
    venv = sb3.common.vec_env.VecFrameStack(venv, kwargs['n_stack'], channels_order=channels_order)

    run = wandb.init(
        project="PPO_Pong",
        config=config,
        sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
        monitor_gym=True,  # auto-upload the videos of agents playing the game
    )

    model = sb3.PPO(
        config["policy_type"], 
        venv, 
        policy_kwargs=config["policy_kwargs"], 
        n_epochs=config['n_epochs'],
        learning_rate=config['learning_rate'],
        batch_size=config["batch_size"],
        vf_coef=config["vf_coef"],
        ent_coef=config["ent_coef"],
        verbose=1, 
        tensorboard_log=f"runs/{run.id}",
    )

    model.learn(
        total_timesteps=config["total_timesteps"],
        callback=WandbCallback(
            model_save_path=f"models/{run.id}",
            verbose=2,
        ),
        eval_freq=kwargs["eval_freq"],
        eval_env=sb3.common.vec_env.VecFrameStack(DummyVecEnv([lambda: Monitor(make_env(config["env_name"], 1, seed=0)())]), kwargs["n_stack"], channels_order=channels_order),
    )

    model_name = "PPO_pong"
    model.save(model_name)
    run.finish()
