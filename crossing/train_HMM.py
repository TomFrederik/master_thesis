import itertools as it

import einops
import numpy as np
import torch
from tqdm import tqdm

from dvae import dVAE

device = 'cuda' if torch.cuda.is_available() else 'cpu'

CONSTANT_ENV = False

data_file = "/home/aric/Desktop/Projects/Master Thesis/ppo_const_env_experience.npz" if CONSTANT_ENV else "/home/aric/Desktop/Projects/Master Thesis/ppo_changing_env_experience.npz"
checkpoint_path = '/home/aric/Desktop/Projects/Master Thesis/Crossing-dVAE/2fxh6iq4/checkpoints/last.ckpt' if CONSTANT_ENV else '/home/aric/Desktop/Projects/Master Thesis/Crossing-dVAE/v31pnq9x/checkpoints/last.ckpt' #TODO
results_file = "/home/aric/Desktop/Projects/Master Thesis/A_const.npz" if CONSTANT_ENV else "/home/aric/Desktop/Projects/Master Thesis/A_changing.npz"
repr_model = dVAE.load_from_checkpoint(checkpoint_path)
repr_model.to(device)

# load data
data = np.load(data_file)
obs = data['obs']
done = data['done']

# split trajectories
stop = np.argwhere(done == 1)
num_trajectories = len(stop)

# for every latent state compute the emission probabilities -> can be reused for every trajectory
num_variables = repr_model.hparams.args.num_variables
codebook_size = repr_model.hparams.args.num_embeddings
latent_size = codebook_size ** num_variables

# create latents
latents = torch.stack([torch.arange(codebook_size, device=device) for _ in range(num_variables)], dim=-1)
latents = torch.zeros((latent_size, codebook_size, num_variables), device=device)
for i, idcs in enumerate(it.product(*(range(codebook_size) for _ in range(num_variables)))):
    latents[i, idcs, torch.arange(num_variables)] = 1
latents = einops.rearrange(latents, 'b c n -> b (c n)')
emission_probs = repr_model.quantize_decode(latents)
emission_probs = einops.rearrange(emission_probs, 'b c h w -> b c (h w)')
print(f"{emission_probs.shape = }")
print(f"{emission_probs.sum(dim=1) = }")
print(f"{emission_probs.max(dim=1)[0].log().sum(dim=1) = }")

# init A to uniform
A = torch.ones((latent_size,latent_size), device=device) / latent_size

# set prior over latent
prior = torch.ones(latent_size, device=device) / latent_size

# init accumulation storage for A
accum_A = torch.zeros_like(A)

n_iter = 5
for k in range(n_iter):
    prev = 0
    for traj in tqdm(range(min(100, num_trajectories))):
        traj_obs = obs[prev:stop[traj,0]]
        traj_obs = traj_obs.reshape(traj_obs.shape[0],-1)
        prev = stop[traj,0]
        if len(traj_obs) > 50: # otherwise I'll run out of cuda memory in the xi_ijt computation
            continue

        traj_emission_probs = torch.stack([emission_probs[:, traj_obs[i], torch.arange(emission_probs.shape[-1])] for i in range(len(traj_obs))], dim=-1)
        traj_emission_probs = torch.exp(torch.log(traj_emission_probs).sum(dim=1)) # is better for stability
        print(f"{traj_emission_probs.shape = }")
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
        # print(f"{alphas[alphas != 0] = }")
        # print(constants)
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
        print(f"{gammas.sum(dim=0) = }")
        xi_ijt =  1/constants[None, None, 1:] * A[...,None] * traj_emission_probs[None,:,1:] * torch.einsum('it, jt -> ijt', alphas[:,:-1], betas[:,1:])
        denominator = xi_ijt.sum(dim=2)
        numerator = denominator.sum(dim=1)
        # print(f"{numerator[numerator == 0] = }")
        accum_A += (denominator / numerator[:,None])

        assert not np.isnan(accum_A.cpu().numpy().sum())
    A = accum_A / num_trajectories
    accum_A = torch.zeros_like(A)
    assert not np.isnan(A.cpu().numpy().sum()), "NaN 1"
# #TODO
print(f"{A = }")
print(f"{prior = }")

np.savez_compressed(results_file, A=A.cpu().numpy(), pi=prior.cpu().numpy())