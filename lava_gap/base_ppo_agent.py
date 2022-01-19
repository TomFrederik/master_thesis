import os
from typing import Optional

import einops
import gym
import stable_baselines3 as sb3
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch
import torch.nn as nn

from wrappers import MultiViewWrapper, DropoutWrapper, MyFullFlatWrapper, MyFullWrapper

class EmbeddingFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Box, embedding_dim: Optional[int] = 32, feature_dim: Optional[int] = 32):

        num_embeddings = len(set(observation_space.sample().flatten()))
        super().__init__(observation_space, feature_dim)

        self.embedding = nn.Embedding(num_embeddings, embedding_dim)

        self.cnn = nn.Sequential(
            nn.Conv2d(embedding_dim, 32, kernel_size=3, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, feature_dim, kernel_size=3, stride=2, padding=0),
            nn.ReLU(),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        b, h, w = observations.shape
        obs = self.embedding(einops.rearrange(observations.int(), 'b h w -> b (h w)'))
        obs = einops.rearrange(obs, 'b (h w) c -> b c h w', h=h, w=w)
        obs = self.cnn(obs)
        obs = einops.rearrange(obs, 'b c h w -> b (c h w)')
        return obs


os.environ["CUDA_LAUNCH_BLOCKING"] = str(1)


env = gym.make('MiniGrid-LavaCrossingS9N3-v0')
# env = MyFullFlatWrapper(env)
env = MyFullWrapper(env)

policy_kwargs = dict(
    features_extractor_class=EmbeddingFeatureExtractor,
    features_extractor_kwargs=dict(embedding_dim=32, feature_dim=32),
)

model = sb3.PPO("MlpPolicy", env, policy_kwargs=policy_kwargs, verbose=1, gamma=0.999, batch_size=64, ent_coef=1)
model.learn(total_timesteps=100000)

obs = env.reset()
for i in range(1000):
    action, _state = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    env.render()
    if done:
      obs = env.reset()
