import argparse

import gym
from gym_minigrid.wrappers import FullyObsWrapper
import numpy as np
import stable_baselines3 as sb3
from tqdm import tqdm

from wrappers import MyFullWrapper, NumSeedsEnvWrappper, StepWrapper

def main(
    num_seeds,
):
    if num_seeds is None:
        num_trajectories = 10000
        suffix = 'all'
    else:
        num_trajectories = 1000 * num_seeds
        suffix = '{}'.format(num_seeds)
        
    model_name = "PPO_all"
    data_file = f"ppo_{suffix}_env_experience"
    model = sb3.PPO.load(model_name)

    # set up environment
    env = gym.make("MiniGrid-SimpleCrossingS9N1-v0")
    env = StepWrapper(env)
    env = FullyObsWrapper(env)
    env = MyFullWrapper(env)
    env = NumSeedsEnvWrappper(env, num_seeds)

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
            obs_list.append(obs)
            obs, rewards, done, info = env.step(action)
            action_list.append(action)
            done_list.append(int(done))

        action, _states = model.predict(obs)
        obs_list.append(obs)
        action_list.append(action)    
    np.savez_compressed(data_file, obs=np.array(obs_list), action=np.array(action_list), done=np.array(done_list))






if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_seeds', type=int, default=None)
    args = parser.parse_args()
    main(**vars(args))