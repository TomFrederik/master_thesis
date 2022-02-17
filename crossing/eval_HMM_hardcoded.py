import argparse
import os

import einops
import numpy as np
import torch
from tqdm import tqdm

from dvae import dVAE


def main(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'{args.constant_env = }')
    
    folder = "/home/aric/Desktop/Projects/Master Thesis"

    data_file = "ppo_const_env_experience.npz" if args.constant_env else "ppo_changing_env_experience.npz"
    results_file = "A_hardcoded_constant.npz" if args.constant_env else "A_hardcoded_changing.npz"
    
    data_file = os.path.join(folder, data_file)
    results_file = os.path.join(folder, results_file)

    # load data
    data = np.load(data_file)
    obs = data['obs']
    done = data['done']

    A = torch.from_numpy(np.load(results_file)['A']).to(device)
    
    prior = torch.zeros(49, device=device, dtype=torch.float64)
    prior[0] = 1

    # split trajectories
    stop = np.argwhere(done == 1)
    num_trajectories = len(stop)

    prev = 0
    for traj in tqdm(range(num_trajectories)):
        traj_obs = torch.from_numpy(obs[prev:stop[traj,0] + 1]).to(device)
        traj_obs = einops.rearrange(traj_obs, 'b h w -> b (h w)')
        prev = stop[traj,0] + 1

        traj_emission_probs = torch.zeros_like(traj_obs)
        traj_emission_probs[traj_obs == 0] = 1
        traj_emission_probs = einops.rearrange(traj_emission_probs, 't d -> d t') # to be consistent
        probs = prior.clone()
        print(f"{A = }")
        print(f"{probs = }")
        for t in range(len(traj_obs)-1):
            probs = probs @ A
            # probs *= traj_emission_probs[:,t+1]
            probs /= probs.sum()
            # print(probs)
            # cross_ent = -(torch.log(probs) * traj_emission_probs[:,t+1])
            cross_ent = (-torch.log(probs[traj_emission_probs[:,t+1] == 1])).sum()
            # cross_ent[traj_emission_probs[:,t+1] == 0] = 0
            # cross_ent = cross_ent.sum()
            prior_ent = -(torch.log(traj_emission_probs[:,t+1]) * traj_emission_probs[:,t+1])
            prior_ent[traj_emission_probs[:,t+1] == 0] = 0
            prior_ent = prior_ent.sum()
            cross_ent_to_last = -(torch.log(traj_emission_probs[:,t]) * traj_emission_probs[:,t+1])
            cross_ent_to_last[traj_emission_probs[:,t+1] == 0] = 0
            cross_ent_to_last = cross_ent_to_last.sum()
            print(f"Step {t+1}: {cross_ent = :.4f}, {prior_ent = :.3f}, {cross_ent_to_last = :.3f}")
        print(probs)
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--constant_env', action='store_true')

    args = parser.parse_args()

    main(args)