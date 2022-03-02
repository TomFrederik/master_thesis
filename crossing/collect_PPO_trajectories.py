import gym
from gym_minigrid.wrappers import FullyObsWrapper
import numpy as np
import stable_baselines3 as sb3
from tqdm import tqdm

from wrappers import MyFullWrapper, DeterministicEnvWrappper, StepWrapper


CONSTANT_ENV = True
normalized_float = False
num_trajectories = 1000 if CONSTANT_ENV else 20000

model_name = "PPO" if CONSTANT_ENV else "PPO_changing"
data_file = "ppo_const_env_experience" if CONSTANT_ENV else "ppo_changing_env_experience"
if normalized_float:
    data_file = data_file + "_normalized_float"
model = sb3.PPO.load(model_name)

# set up environment
env = gym.make("MiniGrid-SimpleCrossingS9N1-v0")
env = StepWrapper(env)
env = FullyObsWrapper(env)
env = MyFullWrapper(env)
if CONSTANT_ENV:
    env = DeterministicEnvWrappper(env)

# Collect trajectories
obs_list = []
action_list = []
done_list = []
for traj in tqdm(range(num_trajectories)):
    obs = env.reset()
    done = False
    done_list.append(int(done))
    while not done:
        action, _states = model.predict(obs)
        if normalized_float:
            obs = (obs - 1.5) / 3
        obs_list.append(obs)
        obs, rewards, done, info = env.step(action)
        action_list.append(action)
        done_list.append(int(done))
    if normalized_float:
        obs = (obs - 1.5) / 3
    obs_list.append(obs)    

np.savez_compressed(data_file, obs=np.array(obs_list), action=np.array(action_list), done=np.array(done_list))