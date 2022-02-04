import numpy as np
import torch
from tqdm import tqdm

from dvae import dVAE

EPS = 1e-10

device = 'cuda' if torch.cuda.is_available() else 'cpu'

CONSTANT_ENV = True

data_file = "ppo_const_env_experience.npz" if CONSTANT_ENV else "ppo_changing_env_experience.npz"
checkpoint_path = 'Crossing-dVAE/3h8l30p7/checkpoints/last.ckpt' if CONSTANT_ENV else 'Crossing-dVAE/{}/checkpoints/last.ckpt'
results_file = "A_const.npz" if CONSTANT_ENV else "A_changing.npz"
repr_model = dVAE.load_from_checkpoint(checkpoint_path)
repr_model.to(device)

# set prior over latent #TODO
prior = torch.ones(1024, device=device) / 1024

# load data
data = np.load(data_file)
obs = data['obs']
done = data['done']
A = np.load(results_file['A'])

# split trajectories
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
    traj_emission_probs += EPS # for stability, otherwise get nan's
    traj_emission_probs /= traj_emission_probs.sum(dim=1)[:,None] # renormalize
    
    # forward
    alpha_0 = prior.clone() * traj_emission_probs[:,0]
    constants = [alpha_0.sum()]
    alphas = [alpha_0/constants[0]]
    for t in range(1,len(traj_obs)):
        new_alpha = alphas[-1] @ A * traj_emission_probs[:,t]
        constants.append(new_alpha.sum())
        new_alpha /= constants[-1]
        alphas.append(new_alpha)
    alphas = torch.stack(alphas, dim=-1)

    # backward
    betas = [torch.ones_like(prior)]
    for t in range(1, len(traj_obs)):
        new_beta = A @ betas[-1] * traj_emission_probs[:,-t]
        new_beta /= constants[-t]
        betas.append(new_beta)
    betas = torch.stack(betas, dim=-1)
    constants = torch.stack(constants, dim=-1)