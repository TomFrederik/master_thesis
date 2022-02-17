import argparse
import itertools as it

import einops
import numpy as np
import torch
from tqdm import tqdm
from torchvision.utils import make_grid
import wandb

from dvae import dVAE


def main(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    data_file = "/home/aric/Desktop/Projects/Master Thesis/ppo_const_env_experience.npz" if args.constant_env else "/home/aric/Desktop/Projects/Master Thesis/ppo_changing_env_experience.npz"
    checkpoint_path = '/home/aric/Desktop/Projects/Master Thesis/Crossing-dVAE/3bk19w36/checkpoints/last.ckpt' if args.constant_env else '/home/aric/Desktop/Projects/Master Thesis/Crossing-dVAE/TODO/checkpoints/last.ckpt' #TODO
    results_file = "/home/aric/Desktop/Projects/Master Thesis/A_const.npz" if args.constant_env else "/home/aric/Desktop/Projects/Master Thesis/A_changing.npz"
    repr_model = dVAE.load_from_checkpoint(checkpoint_path)
    repr_model.to(device)

    # load data
    data = np.load(data_file)
    obs = data['obs']
    done = data['done']

    A = torch.from_numpy(np.load(results_file)['A']).to(device)
    prior = torch.from_numpy(np.load(results_file)['pi']).to(device)
    # split trajectories
    stop = np.argwhere(done == 1)
    num_trajectories = len(stop)
    
    num_variables = repr_model.hparams.args.num_variables
    codebook_size = repr_model.hparams.args.num_embeddings
    latent_size = codebook_size ** num_variables

    # for every latent state compute the emission probabilities -> can be reused for every trajectory
    latents = torch.stack([torch.arange(codebook_size, device=device) for _ in range(num_variables)], dim=-1)
    latents = torch.zeros((latent_size, codebook_size, num_variables), device=device)
    for i, idcs in enumerate(it.product(*(range(codebook_size) for _ in range(num_variables)))):
        latents[i, idcs, torch.arange(num_variables)] = 1
    latents = einops.rearrange(latents, 'b c n -> b (n c)')
    emission_probs = repr_model.quantize_decode(latents)
    emission_probs = einops.rearrange(emission_probs, 'b c h w -> b c (h w)')

    # how do I evaluate A?
    # start from start distribution given by VAE, apply A 12 times. 
    # need to multiply with emission probability
    # Cross entropy(p, q) = Entropy(p) + KL(p|q)

    # init logger
    config = dict(
        constant_env=args.constant_env
    )
    wandb.init(project="Eval-Transition-Matrix", config=config)


    prev = 0
    for traj in tqdm(range(num_trajectories)):
        traj_obs_indices = torch.from_numpy(obs[prev:stop[traj,0]+1]).to(device)
        traj_obs = torch.nn.functional.one_hot(traj_obs_indices, 4).float()
        traj_obs = einops.rearrange(traj_obs, 'b h w c -> b c h w')
        reconstructed_originals = repr_model.reconstruct_only(traj_obs).cpu().float()
        *_, vae_logits = repr_model.encode_only(traj_obs)
        print(f"{vae_logits.shape = }")
        summed_logits = vae_logits[:,0]
        
        for i in range(1, vae_logits.shape[1]):
            summed_logits = summed_logits[...,None] + einops.rearrange(vae_logits[:,i], f'a b -> a {" ".join(str(1) for _ in range(i))} b')
        vae_logits = summed_logits
        vae_logits = einops.rearrange(vae_logits, 'b ... -> b (...)')
        vae_probs = torch.softmax(vae_logits, dim=-1)
        
        # reshape for indexing
        traj_obs_indices = traj_obs_indices.reshape(traj_obs_indices.shape[0],-1)
        traj_emission_probs = torch.stack([emission_probs[:, traj_obs_indices[i], torch.arange(emission_probs.shape[-1])] for i in range(len(traj_obs_indices))], dim=-1)
        traj_emission_probs = einops.rearrange(traj_emission_probs, 'codebook_size num_pixels time -> time num_pixels codebook_size')
        traj_emission_probs = torch.log(traj_emission_probs)
        traj_emission_probs = traj_emission_probs.sum(dim=1)
        traj_emission_probs = torch.exp(traj_emission_probs)
        print(f"{traj_emission_probs.shape = }")
        
        probs = prior.clone()
        probs = torch.ones_like(probs)
        probs /= probs.sum()
        probs *= traj_emission_probs[0]
        probs /= probs.sum()
        images = []
        for t in range(len(traj_obs)-1):
            
            print(torch.where(probs == 0))
            probs = probs @ A
            print(probs)
            probs = probs * traj_emission_probs[t+1]
            print(probs)
            print(probs.shape)
            print(torch.where(probs == 0))
            print(torch.where(traj_emission_probs[t+1] == 0))
            # print(torch.where(A == 0))
            probs /= probs.sum()
            print(torch.where(probs == 0))
            # print(probs[probs.isnan()])
            # print(torch.where(probs == 0))
            # print(torch.where(vae_probs[t+1] == 0))
            cross_ent = -(torch.log(probs) * vae_probs[t+1])
            cross_ent[vae_probs[t+1] == 0] = 0
            cross_ent = cross_ent.sum()
            vae_ent = -(torch.log(vae_probs[t+1]) * vae_probs[t+1]).sum()
            cross_ent_to_last = -(torch.log(vae_probs[t]) * vae_probs[t+1]).sum()
            print(f"{cross_ent:.4f}, {vae_ent:.4f}, {cross_ent_to_last:.4f}")
            # marginals = einops.rearrange(probs[None], f'b ({" ".join("w" for _ in range(codebook_size))}) -> b {" ".join("w" for _ in range(codebook_size))}', w=num_variables)
            # print(f"{marginals.shape = }")
            # marginals = marginals.stack()
            # marginals = torch.stack([marginals.sum(dim=1), marginals.sum(dim=2)], dim=1)
            # print(f"{marginals.shape = }")
            # marginals = einops.rearrange(marginals, 'b n c -> b (n c)')
            # print(f"{marginals.shape = }")
            # image_probs = repr_model.quantize_decode(marginals)
            # image_probs = einops.rearrange(image_probs, 'b c h w -> (b h w) c')
            # image = torch.multinomial(image_probs, 1).float()
            # image = einops.rearrange(image, '(h w) c -> (c h) w', h=7, w=7)
            # images.append(image)
            raise ValueError
        images = torch.stack(images, dim=0).cpu()
        images = torch.stack([torch.from_numpy(obs[prev:stop[traj,0]+1]), reconstructed_originals,  torch.cat([torch.zeros((1,7,7)), images], dim=0)], dim=1).reshape((-1, 1, *images.shape[1:]))

        # log images to tensorboard
        wandb.log({'Prediction': wandb.Image(make_grid(images, nrow=3))})
        
        prev = stop[traj,0]+1
        if traj > 10:
            raise ValueError()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--constant_env', action='store_true')

    args = parser.parse_args()

    main(args)