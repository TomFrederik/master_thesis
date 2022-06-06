import argparse

import gym
from gym_minigrid.wrappers import FullyObsWrapper
import numpy as np
import stable_baselines3 as sb3
from tqdm import tqdm

from wrappers import MyFullWrapper, NumSeedsEnvWrappper, StepWrapper

def main(
    num_seeds,
    max_traj_length,
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
    traj = 0
    while traj < num_trajectories:
        obs = env.reset()
        done = False
        done_candidates = [int(done)] 
        obs_candidates = []
        action_candidates = []
        while not done:
            action, _states = model.predict(obs)
            obs_candidates.append(obs)
            obs, rewards, done, info = env.step(action)
            action_candidates.append(action)
            done_candidates.append(int(done))

        action, _states = model.predict(obs)
        obs_candidates.append(obs)
        action_candidates.append(action)    
        if len(done_candidates) > max_traj_length:
            continue
        else:
            traj += 1
            done_list.extend(done_candidates)
            obs_list.extend(obs_candidates)
            action_list.extend(action_candidates)
    np.savez_compressed(data_file, obs=np.array(obs_list), action=np.array(action_list), done=np.array(done_list))






if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_seeds', type=int, default=None)
    parser.add_argument('--max_traj_length', type=int, default=20)
    args = parser.parse_args()
    main(**vars(args))