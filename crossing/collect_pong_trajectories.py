import argparse

import gym
from gym_minigrid.wrappers import FullyObsWrapper
import numpy as np
import stable_baselines3 as sb3
from tqdm import tqdm

from wrappers import MyFullWrapper, NumSeedsEnvWrappper, StepWrapper
from stable_baselines3.common.atari_wrappers import EpisodicLifeEnv, NoopResetEnv
from stable_baselines3.common.vec_env.dummy_vec_env import DummyVecEnv
from stable_baselines3.common.vec_env import VecFrameStack
from ppo_pong import CropGrayWrapper, make_env

def main(
    num_trajectories,
):
        
    model_name = "PPO_pong"
    data_file = f"ppo_pong_experience"
    model = sb3.PPO.load(model_name)

    # set up environment
    env = gym.make("Pong-v0")
    venv= [make_env("Pong-v0", rank=0, seed=0)]
    venv = DummyVecEnv(venv)
    venv = VecFrameStack(venv, 4, channels_order="first")

    # Collect trajectories
    obs_list = []
    action_list = []
    done_list = []
    reward_list = []
    traj = 0

    import matplotlib.pyplot as plt

    while traj < num_trajectories:
        obs = venv.reset()
        done = False
        done_candidates = [int(done)] 
        obs_candidates = []
        action_candidates = []
        reward_candidates = []
        while not done:
            action, _states = model.predict(obs)
            obs_candidates.append(obs[:, 0]) # (1, 4, 84, 84) -> (1, 84, 84)
            obs, rewards, done, info = venv.step(action)
            action_candidates.append(action)
            done_candidates.append(int(done))
            plt.imshow(obs_candidates[-1][0], cmap='gray')
            plt.show()
            plt.close()
            
        action, _states = model.predict(obs)
        obs_candidates.append(obs)
        action_candidates.append(action)    
        reward_candidates.append(rewards)
        traj += 1
        done_list.extend(done_candidates)
        obs_list.extend(obs_candidates)
        action_list.extend(action_candidates)
        reward_list.extend(reward_candidates)
    np.savez_compressed(data_file, obs=np.array(obs_list), action=np.array(action_list), rewards=np.array(reward_list), done=np.array(done_list))






if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_trajectories', type=int, default=1000)
    args = parser.parse_args()
    main(**vars(args))