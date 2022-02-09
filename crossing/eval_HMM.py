import einops
import numpy as np
import torch
from tqdm import tqdm

from dvae import dVAE

device = 'cuda' if torch.cuda.is_available() else 'cpu'

CONSTANT_ENV = True

data_file = "/home/aric/Desktop/Projects/Master Thesis/ppo_const_env_experience.npz" if CONSTANT_ENV else "/home/aric/Desktop/Projects/Master Thesis/ppo_changing_env_experience.npz"
checkpoint_path = '/home/aric/Desktop/Projects/Master Thesis/Crossing-dVAE/2fxh6iq4/checkpoints/last.ckpt' if CONSTANT_ENV else '/home/aric/Desktop/Projects/Master Thesis/Crossing-dVAE/TODO/checkpoints/last.ckpt' #TODO
results_file = "/home/aric/Desktop/Projects/Master Thesis/A_const.npz" if CONSTANT_ENV else "/home/aric/Desktop/Projects/Master Thesis/A_changing.npz"
repr_model = dVAE.load_from_checkpoint(checkpoint_path)
repr_model.to(device)

# load data
data = np.load(data_file)
obs = data['obs']
done = data['done']

A = torch.from_numpy(np.load(results_file)['A']).to(device)

# split trajectories
stop = np.argwhere(done == 1)
num_trajectories = len(stop)

# for every latent state compute the emission probabilities -> can be reused for every trajectory
latents = torch.arange(1024, device=device)
latents = torch.nn.functional.one_hot(latents).float()
print(latents.shape)
emission_probs = repr_model.quantize_decode(latents).reshape(1024,4,-1)


# how do I evaluate A?
# start from start distribution given by VAE, apply A 12 times. 
# need to multiply with emission probability
# Cross entropy(p, q) = Entropy(p) + KL(p|q)

prev = 0
for traj in tqdm(range(num_trajectories)):
    traj_obs_indices = torch.from_numpy(obs[prev:stop[traj,0]]).to(device)
    traj_obs = torch.nn.functional.one_hot(traj_obs_indices, 4).float()
    traj_obs = einops.rearrange(traj_obs, 'b h w c -> b c h w')
    *_, logits = repr_model.encode_only(traj_obs)
    # multiply all variables to form a single variable
    logits = logits.sum(dim=1)
    # softmax to normalize
    prior = torch.softmax(logits, dim=-1)

    # reshape for indexing
    traj_obs_indices = traj_obs_indices.reshape(traj_obs_indices.shape[0],-1)
    traj_emission_probs = torch.stack([emission_probs[:, traj_obs_indices[i], torch.arange(emission_probs.shape[-1])] for i in range(len(traj_obs_indices))], dim=-1)
    traj_emission_probs = torch.softmax(torch.log(traj_emission_probs).sum(dim=1), dim=1) # is better for stability
    traj_emission_probs = einops.rearrange(traj_emission_probs, 'd t -> t d')
    

    # print(traj_obs)    
    probs = prior[0]
    for t in range(len(traj_obs)-1):
        probs = probs @ A
        probs *= traj_emission_probs[t+1]
        probs /= probs.sum()
        cross_ent = -(torch.log(probs) * prior[t+1]).sum()
        prior_ent = -(torch.log(prior[t+1]) * prior[t+1]).sum()
        cross_ent_to_last = -(torch.log(prior[t]) * prior[t+1]).sum()
        print(f"Step {t+1}: {cross_ent = :.3f}, {prior_ent = :.3f}, {cross_ent_to_last = :.3f}")

        image_probs = repr_model.quantize_decode(probs[None])
        image_probs = einops.rearrange(image_probs, 'b c h w -> (b h w) c')
        image = torch.multinomial(image_probs, 1)
        image = einops.rearrange(image, '(h w) c -> c h w', h=7, w=7)
        print('correct_image:', obs[prev:stop[traj,0]][t])
        print('predicted image:', image)
        print('\n\n')

    prev = stop[traj,0]
    raise ValueError()