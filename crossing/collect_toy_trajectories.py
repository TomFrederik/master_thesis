import argparse

import gym
from gym_minigrid.wrappers import FullyObsWrapper
import numpy as np
import stable_baselines3 as sb3
from tqdm import tqdm
import h5py

from wrappers import MyFullWrapper, StepWrapper

def main(
    num_trajectories
):
    
    f = h5py.File('toy_data.hdf5', 'w') # change this to 'a' to prevent overwrite?
    f.create_group('done')
    f.create_group('obs')
    f.create_group('action')
    f.create_group('reward')
    
    model_name = "toy_ppo"
    model = sb3.PPO.load(model_name)

    # set up environment
    env = gym.make("MiniGrid-SimpleCrossingS9N1-v0")
    env = StepWrapper(env)
    env = FullyObsWrapper(env)
    env = MyFullWrapper(env)
    
    # running stats helpers
    ctr = 0
    first_moment = 0
    second_moment = 0

    # Collect trajectories
    obs_list = []
    action_list = []
    done_list = []
    reward_list = []
    for traj in tqdm(range(num_trajectories)):
        obs = env.reset()
        done = False
        done_candidates = [int(done)] 
        obs_candidates = []
        action_candidates = []
        reward_candidates = []
        while not done:
            action, _states = model.predict(obs)
            obs_candidates.append(obs)
            obs, reward, done, info = env.step(action)
            action_candidates.append(action)
            reward_candidates.append(reward)
            done_candidates.append(int(done))

        action, _states = model.predict(obs)
        obs_candidates.append(obs)
        action_candidates.append(action)    


        done_list.extend(done_candidates)
        obs_list.extend(obs_candidates)
        action_list.extend(action_candidates)
        reward_list.extend(reward_candidates)
        
        # update running stats
        ctr += len(obs_candidates) * 7 * 7
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
    parser.add_argument('--num_trajectories', type=int, default=10000)
    args = parser.parse_args()
    main(**vars(args))