import itertools as it
import os

import einops
import numpy as np
import torch
from tqdm import tqdm

device = 'cuda' if torch.cuda.is_available() else 'cpu'

CONSTANT_ENV = False

folder = "/home/aric/Desktop/Projects/Master Thesis"

data_file = "ppo_const_env_experience.npz" if CONSTANT_ENV else "ppo_changing_env_experience.npz"
results_file = "A_hardcoded_constant.npz" if CONSTANT_ENV else "A_hardcoded_changing.npz"

data_file = os.path.join(folder, data_file)
results_file = os.path.join(folder, results_file)

# load data
data = np.load(data_file)
obs = data['obs']
done = data['done']

# split trajectories
stop = np.argwhere(done == 1)
num_trajectories = len(stop)


# create latents
# init A to uniform
A = torch.ones((49,49), device=device, dtype=torch.float64) / 49

# set prior over latent
prior = torch.ones(49, device=device, dtype=torch.float64) / 49

# init accumulation storage for A
accum_A = torch.zeros_like(A)

n_iter = 1
for k in range(n_iter):
    prev = 0
    for traj in tqdm(range(num_trajectories)):
        traj_obs = torch.from_numpy(obs[prev:stop[traj,0] + 1]).to(device)
        traj_obs = einops.rearrange(traj_obs, 'b h w -> b (h w)')

        traj_emission_probs = torch.zeros_like(traj_obs)
        traj_emission_probs[traj_obs == 0] = 1
        traj_emission_probs = einops.rearrange(traj_emission_probs, 't d -> d t') # to be consistent

        prev = stop[traj,0] + 1
        
        # forward
        alpha_0 = prior.clone() * traj_emission_probs[:,0]
        constants = [alpha_0.sum()]
        alphas = [alpha_0/constants[0]]
        for t in range(1,len(traj_obs)):
            new_alpha = (alphas[-1] @ A) * traj_emission_probs[:,t]
            constants.append(new_alpha.sum())
            new_alpha /= constants[-1]
            alphas.append(new_alpha)
        alphas = torch.stack(alphas, dim=-1)
        
        # backward
        betas = [torch.ones_like(prior)]
        for t in range(1, len(traj_obs)):
            new_beta = A @ (betas[-1] * traj_emission_probs[:,-t])
            new_beta = new_beta / constants[-t]
            betas.append(new_beta)
        betas = torch.stack(betas[::-1], dim=-1)
        constants = torch.stack(constants, dim=-1)
        gammas = alphas * betas 
        prior = gammas[:,0] / gammas[:,0].sum() # update prior
        xi_ijt =  1/constants[None, None, 1:] * A[...,None] * traj_emission_probs[None,:,1:] * alphas[:,None,:-1] * betas[None,:,1:]
        denominator = xi_ijt.sum(dim=2)
        accum_A += denominator
    update_A = accum_A / accum_A.sum(dim=1)[:,None]
    A[update_A.sum(dim=1) > 0] = update_A[update_A.sum(dim=1) > 0]
    print(f"{A = }")
    accum_A = torch.zeros_like(A)
    assert not np.isnan(A.cpu().numpy().sum()), "NaN 1"

np.savez_compressed(results_file, A=A.cpu().numpy(), pi=prior.cpu().numpy())