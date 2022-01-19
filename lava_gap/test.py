import os

import gym
import numpy as np

from utils import display_helper
from wrappers import MultiViewWrapper, DropoutWrapper

env = gym.make('MiniGrid-LavaCrossingS9N3-v0')
env = MultiViewWrapper(env, percentages=[0.5, 0.5, 0.5])
env = DropoutWrapper(env, dropout_p=0.1)
obs = env.reset()

display_helper(obs, save_path='./lava_gap/lava_all_views.pdf')