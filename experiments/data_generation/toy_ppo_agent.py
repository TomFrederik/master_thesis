import os
from typing import Optional

import einops
import gym
import stable_baselines3 as sb3
import torch
import torch.nn as nn
import wandb
from einops.layers.torch import Rearrange
from gym_minigrid.envs import SimpleCrossingEnv
from gym_minigrid.wrappers import FullyObsWrapper
from stable_baselines3.common import env_checker
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from wandb.integration.sb3 import WandbCallback

from src.common.wrappers import (DropoutWrapper, MultiViewWrapper, MyFullFlatWrapper,
                      MyFullWrapper, NumSeedsEnvWrappper, StepWrapper)


class EmbeddingFeatureExtractor(BaseFeaturesExtractor):
    def __init__(
        self, 
        observation_space: gym.spaces.Box, 
        embedding_dim: Optional[int] = 32, 
        feature_dim: Optional[int] = 32,
        num_channels: Optional[int] = 32,
    ):

        super().__init__(observation_space, feature_dim)

        self.num_embeddings = 4
        embedding_dim = 4

        # self.embedding = nn.Embedding(self.num_embeddings, embedding_dim)

        self.image_conv = nn.Sequential(
            nn.Conv2d(embedding_dim, num_channels, kernel_size=7),
            nn.ReLU(),
            Rearrange('b c h w -> b (c h w)'),
            nn.Linear(num_channels, feature_dim),
            nn.ReLU(),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        b, h, w = observations.shape
        # one-hot
        obs = nn.functional.one_hot(observations.long(), num_classes=4).float()
        obs = einops.rearrange(obs, 'b h w c -> b c h w')
        # obs = self.embedding(einops.rearrange(observations.int(), 'b h w -> b (h w)'))
        # obs = einops.rearrange(obs, 'b (h w) c -> b c h w', h=h, w=w)
        obs = self.image_conv(obs)
        return obs




# os.environ["WANDB_MODE"] = "offline"
# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

config = {
    "policy_type": "MlpPolicy",
    "total_timesteps": 500000,
    "env_name": "MiniGrid-SimpleCrossingS9N1-v0",
    "policy_kwargs": dict(
        features_extractor_class=EmbeddingFeatureExtractor,
    ),
    "batch_size": 256,
    "vf_coef": 0.5,
    "ent_coef": 0.01,
    "embedding_dim": 128,
    "num_channels": 128,
    'feature_dim': 512,
    "network_dim": 512,
    'n_epochs':4,
    'learning_rate':0.0001,
}

config['policy_kwargs']['features_extractor_kwargs'] = dict(embedding_dim=config['embedding_dim'], feature_dim=config['feature_dim'], num_channels=config['num_channels'])
config['policy_kwargs']['net_arch'] = [dict(pi=[config['feature_dim'], config['network_dim']], vf=[config['feature_dim'], config['network_dim']])]

env = gym.make("MiniGrid-SimpleCrossingS9N1-v0")
# env = gym.make(config['env_name'])
env_checker.check_env(env)
if not isinstance(env, SimpleCrossingEnv):
    raise TypeError
env = StepWrapper(env)
env = FullyObsWrapper(env)
env = MyFullWrapper(env)
obs = env.reset()
done = False
while not done:
    print(obs)
    obs, reward, done, info = env.step(env.action_space.sample())
    print(reward)
print(obs)
obs = env.reset()
print(obs)
raise NotImplementedError
run = wandb.init(
    project="MiniGrid-Crossing",
    config=config,
    sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
    # monitor_gym=True,  # auto-upload the videos of agents playing the game
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

model_name = "PPO_toy"
model.save(model_name)
run.finish()
