import os
from typing import Optional

import einops
from einops.layers.torch import Rearrange
import gym
from gym_minigrid.wrappers import FullyObsWrapper
import stable_baselines3 as sb3
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch
import torch.nn as nn
import wandb
from wandb.integration.sb3 import WandbCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env.base_vec_env import VecEnvWrapper
import numpy as np
from torchvision.transforms import Resize, Grayscale

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
        embedding_dim: Optional[int] = 32, 
        feature_dim: Optional[int] = 32,
        num_channels: Optional[int] = 32,
    ):

        super().__init__(observation_space, feature_dim)

        input_shape = observation_space.shape

        d = num_channels
        k = 3
        stride = 2
        conv1_shape = conv_out_shape(input_shape[1:], 0, k, stride)
        conv2_shape = conv_out_shape(conv1_shape, 0, k, stride)
        conv3_shape = conv_out_shape(conv2_shape, 0, k, stride)
        conv4_shape = conv_out_shape(conv3_shape, 0, k, stride)
        self.conv_shape = (d, *conv4_shape)
        print(f"{self.conv_shape = }")

        self.image_conv = nn.Sequential(
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
        
        print("\nImage conv net:")
        print(self.image_conv)
        
    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        obs = self.image_conv(observations)
        return obs

class CropGrayWrapper(VecEnvWrapper):
    def __init__(self, venv: VecEnvWrapper):
        super().__init__(venv)
        self.venv = venv
        self.trafo_list = [
            lambda x: x[:, :,35:-25],
            lambda x: x/228,
            Resize((84,84)),
            Grayscale(),
        ]
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(1, 84, 84), dtype=np.float32)
        
    def trafos(self, x):
        for trafo in self.trafo_list:
            x = trafo(x)
        return x
        
    def observation(self, observation):
        observation = torch.from_numpy(einops.rearrange(observation, 'b h w c -> b c h w')).float()
        observation = self.trafos(observation).numpy()
        return observation
    
    def reset(self):
        return self.observation(self.venv.reset())

    def step_wait(self):
        out = self.venv.step_wait()
        return (self.observation(out[0]), *out[1:])

config = {
    "constant_env":False,
    "policy_type": "MlpPolicy",
    "total_timesteps": 500000,
    "env_name": "Pong-v0",
    "policy_kwargs": dict(
        features_extractor_class=ConvFeatureExtractor,
    ),
    "batch_size": 256,
    "vf_coef": 0.5,
    "ent_coef": 0.01,
    "embedding_dim": 64,
    "num_channels": 16,
    'feature_dim': 128,
    "network_dim": 512,
    'n_epochs':4,
    'learning_rate':0.0001,
}



config['policy_kwargs']['features_extractor_kwargs'] = dict(embedding_dim=config['embedding_dim'], feature_dim=config['feature_dim'], num_channels=config['num_channels'])
config['policy_kwargs']['net_arch'] = [dict(pi=[config['feature_dim'], config['network_dim']], vf=[config['feature_dim'], config['network_dim']])]



n_stack = 4
channels_order = "first"

venv = make_vec_env(config['env_name'], n_envs=2)
venv = sb3.common.vec_env.VecFrameStack(venv, n_stack, channels_order=channels_order)
venv = CropGrayWrapper(venv)

model = ConvFeatureExtractor(venv.observation_space, config['embedding_dim'], config['feature_dim'], config['num_channels'])

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
)

model_name = "PPO" if config['constant_env'] else "PPO_all"
model.save(model_name)
run.finish()