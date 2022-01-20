import os
from typing import Optional

import einops
import gym
from gym_minigrid.wrappers import FullyObsWrapper
import stable_baselines3 as sb3
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch
import torch.nn as nn
import wandb
from wandb.integration.sb3 import WandbCallback

from wrappers import MultiViewWrapper, DropoutWrapper, MyFullFlatWrapper, MyFullWrapper

class EmbeddingFeatureExtractor(BaseFeaturesExtractor):
    def __init__(
        self, 
        observation_space: gym.spaces.Box, 
        embedding_dim: Optional[int] = 32, 
        feature_dim: Optional[int] = 32,
        num_channels: Optional[int] = 32,
    ):

        num_embeddings = len(set(observation_space.sample().flatten()))
        super().__init__(observation_space, feature_dim)

        self.embedding = nn.Embedding(num_embeddings, embedding_dim)

        self.image_conv = nn.Sequential(
            nn.Conv2d(embedding_dim, num_channels, kernel_size=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(num_channels, 2*num_channels, kernel_size=2),
            nn.ReLU(),
            nn.Conv2d(2*num_channels, 4*num_channels, kernel_size=2),
            nn.ReLU()
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        b, h, w = observations.shape
        obs = self.embedding(einops.rearrange(observations.int(), 'b h w -> b (h w)'))
        obs = einops.rearrange(obs, 'b (h w) c -> b c h w', h=h, w=w)
        obs = self.image_conv(obs)
        obs = einops.rearrange(obs, 'b c h w -> b (c h w)')
        return obs

# os.environ["WANDB_MODE"] = "offline"

config = {
    "policy_type": "MlpPolicy",
    "total_timesteps": 1000000,
    "env_name": "MiniGrid-SimpleCrossingS9N1-v0",
    "policy_kwargs": dict(
        features_extractor_class=EmbeddingFeatureExtractor,
    ),
    "batch_size": 256,
    "vf_coef": 0.5,
    "ent_coef": 0.01,
    "embedding_dim": 128,
    "num_channels": 128,
    "network_dim": 128,
    'n_epochs':4,
    'learning_rate':0.0001,
}
config['policy_kwargs']['features_extractor_kwargs'] = dict(embedding_dim=config['embedding_dim'], feature_dim=4*config['num_channels'], num_channels=config['num_channels'])
config['policy_kwargs']['net_arch'] = [dict(pi=[512, config['network_dim']], vf=[512, config['network_dim']])]

env = gym.make(config['env_name'])
env = MyFullWrapper(env)

run = wandb.init(
    project="MiniGrid-Crossing",
    config=config,
    sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
    monitor_gym=True,  # auto-upload the videos of agents playing the game
)

model = sb3.PPO(
    config["policy_type"], 
    env, 
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
run.finish()