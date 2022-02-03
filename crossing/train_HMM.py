import numpy as np
import torch
from tqdm import tqdm

from dvae import dVAE


device = 'cuda' if torch.cuda.is_available() else 'cpu'

checkpoint_path = 'Crossing-dVAE/3h8l30p7/checkpoints/last.ckpt'
repr_model = dVAE.load_from_checkpoint(checkpoint_path)
repr_model.to(device)

# init A to uniform
A = torch.ones((1024,1024), device=device) / 1024

# set prior over latent #TODO
prior = torch.ones(1024, device=device) / 1024

# load data
data = np.load('ppo_const_env_experience.npz')
obs = data['obs']
done = data['done']

# get first trajectory
print(obs.shape)
stop = np.argwhere(done == 1)
num_trajectories = len(stop)

# for every latent state compute the emission probabilities -> can be reused for every trajectory
latents = torch.arange(1024, device=device)
latents = torch.nn.functional.one_hot(latents).float()
emission_probs = repr_model.quantize_decode(latents).reshape(1024,4,-1)

prev = 0
for traj in tqdm(range(num_trajectories)):
    traj_obs = obs[prev:stop[traj,0]]
    traj_obs = traj_obs.reshape(traj_obs.shape[0],-1)
    prev = stop[traj,0]
    
    traj_emission_probs = torch.stack([emission_probs[:, traj_obs[i], torch.arange(emission_probs.shape[-1])] for i in range(len(traj_obs))], dim=-1)
    traj_emission_probs = torch.prod(traj_emission_probs, dim=1)
    
    # forward
    alpha_0 = prior.clone() * traj_emission_probs[:,0]
    alphas = [alpha_0]
    for t in range(1,len(traj_obs)):
        new_alpha = alphas[-1] @ A * traj_emission_probs[:,t]
        alphas.append(new_alpha)
    alphas = torch.stack(alphas, dim=-1)

    # backward
    betas = [torch.ones_like(prior)]
    for t in range(1, len(traj_obs)):
        new_beta = A @ betas[-1] * traj_emission_probs[:,-t]
        betas.append(new_beta)
    betas = torch.stack(betas, dim=-1)

    # update A
    A = torch.sum(A[...,None] * traj_emission_probs[None,...] * torch.einsum('it, jt -> ijt', alphas, betas), dim=-1)
    
    # renormalize
    A = A / torch.sum(A, dim=0)[None,:]

print(A)
