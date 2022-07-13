import argparse

import gym
import numpy as np
import stable_baselines3 as sb3
from tqdm import tqdm

from stable_baselines3.common.vec_env.dummy_vec_env import DummyVecEnv
from stable_baselines3.common.vec_env import VecFrameStack
from ppo_pong import make_env
import h5py

def main(
    num_trajectories,
):

    f = h5py.File('../pong_data.hdf5', 'w') # change this to 'a' to prevent overwrite?
    f.create_group('done')
    f.create_group('obs')
    f.create_group('action')
    f.create_group('reward')

    model_name = "PPO_pong"
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
    
    # running stats helpers
    ctr = 0
    first_moment = 0
    second_moment = 0

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

        # update running stats
        ctr += len(obs_candidates) * 84 * 84
        first_moment += np.sum(np.array(obs_candidates))
        second_moment += np.sum(np.array(obs_candidates) ** 2)
        
        f['done'].create_dataset(f'traj_{traj}', data=np.array(done_candidates))
        f['reward'].create_dataset(f'traj_{traj}', data=np.array(reward_candidates))
        f['obs'].create_dataset(f'traj_{traj}', data=np.array(obs_candidates))
        f['action'].create_dataset(f'traj_{traj}', data=np.array(action_candidates))

    f['obs'].attrs.create('mean', first_moment / ctr)
    f['obs'].attrs.create('std', np.sqrt(second_moment / ctr - first_moment ** 2 / ctr ** 2))
    f.close()    
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_trajectories', type=int, default=1000)
    args = parser.parse_args()
    main(**vars(args))