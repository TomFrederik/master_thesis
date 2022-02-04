import os
from typing import Optional

import einops
from einops.layers.torch import Rearrange
import gym
from gym_minigrid.wrappers import FullyObsWrapper
import numpy as np
import stable_baselines3 as sb3
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch
import torch.nn as nn
from tqdm import tqdm
import wandb
from wandb.integration.sb3 import WandbCallback

from wrappers import MultiViewWrapper, DropoutWrapper, MyFullFlatWrapper, MyFullWrapper, DeterministicEnvWrappper, StepWrapper


CONSTANT_ENV = False
num_trajectories = 1000 if CONSTANT_ENV else 100000

model_name = "PPO" if CONSTANT_ENV else "PPO_changing"
data_file = "ppo_const_env_experience" if CONSTANT_ENV else "ppo_changing_env_experience"
model = sb3.PPO.load("PPO_changing")

# set up environment
env = gym.make("MiniGrid-SimpleCrossingS9N1-v0")
env = StepWrapper(env)
env = FullyObsWrapper(env)
env = MyFullWrapper(env)
if CONSTANT_ENV:
    env = DeterministicEnvWrappper(env)

# Collect trajectories
obs_list = []
done_list = []
for traj in tqdm(range(num_trajectories)):
    obs = env.reset()
    done = False
    while not done:
        action, _states = model.predict(obs)
        obs, rewards, done, info = env.step(action)
        obs_list.append(obs)
        done_list.append(int(done))

np.savez_compressed(data_file, obs=np.array(obs_list), done=np.array(done_list))