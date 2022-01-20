import os

import gym
import numpy as np

from utils import display_helper
from wrappers import MultiViewWrapper, DropoutWrapper

env = gym.make('MiniGrid-SimpleCrossingS9N3-v0')
print(f"{env.observation_space = }")
env = MultiViewWrapper(env, percentages=[0.5, 0.5, 0.5])
print(f"{env.observation_space = }")
env = DropoutWrapper(env, dropout_p=0.1)
print(f"{env.observation_space.sample().shape = }")
obs = env.reset()

display_helper(obs, save_path='./crossing/crossing_all_views.png')