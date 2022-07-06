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
import h5py

def main(
    num_trajectories,
):

    f = h5py.File('pong_data.hdf5', 'a')
    try:
        f.create_dataset('done', dtype=int, chunks=(0,))
        f.create_dataset('obs', dtype=float, chunks=(0,84,84))
        f.create_dataset('action', dtype=int, chunks=(0,))
        f.create_dataset('reward', dtype=float, chunks=(0,))
    except:
        pass
        # dset = h5py.Group.create_dataset("trajs")

    model_name = "PPO_pong"
    data_file = "ppo_pong_experience"
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

    for traj in tqdm(range(num_trajectories)):
        obs = venv.reset()
        done = False
        done_candidates = [int(done)] 
        obs_candidates = []
        action_candidates = []
        reward_candidates = [np.zeros((1,), dtype=np.float32)]
        while not done:
            action, _states = model.predict(obs)
            obs_candidates.append(obs[0, 0]) # (1, 4, 84, 84) -> (84, 84)
            obs, reward, done, info = venv.step(action)
            action_candidates.append(action)
            reward_candidates.append(reward)
            done_candidates.append(int(done))
            
        action, _states = model.predict(obs)
        obs_candidates.append(obs[0, 0])
        action_candidates.append(action)    

        done_list.extend(done_candidates)
        obs_list.extend(obs_candidates)
        action_list.extend(action_candidates)
        reward_list.extend(reward_candidates)

    
        f['done'].resize((f['done'].shape[0] + len(done_candidates),))
        f['reward'].resize((f['done'].shape[0] + len(done_candidates),))
        f['action'].resize((f['done'].shape[0] + len(done_candidates),))
        f['obs'].resize(f['done'].shape[0] + len(done_candidates), axis=0)
    
        f['done'][-len(done_candidates):] = np.array(done_candidates)
        f['reward'][-len(done_candidates):] = np.array(reward_candidates)
        f['action'][-len(done_candidates):] = np.array(done_candidates)
        f['obs'][-len(done_candidates):] = np.array(obs_candidates)

        print(f['done'].shape)
        print(f['reward'].shape)
        print(f['action'].shape)
        print(f['obs'].shape)
        
    f.close()    
    
    # np.savez_compressed(data_file, obs=np.array(obs_list), action=np.array(action_list), rewards=np.array(reward_list), done=np.array(done_list))
    # data = np.load(data_file+".npz")





if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_trajectories', type=int, default=1000)
    args = parser.parse_args()
    main(**vars(args))