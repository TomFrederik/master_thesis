import einops
import numpy as np
import torch
from tqdm import tqdm

from dvae import dVAE

import matplotlib.pyplot as plt



# extended logarithm
def eln(x):
    out = torch.empty_like(x)
    out[x!=0] = torch.log(x[x!=0])
    out[x==0] = torch.nan
    return out

# extended exponential function
def eexp(x):
    out = torch.empty_like(x)
    out[x!=torch.nan] = torch.exp(out[x!=torch.nan])
    out[x==torch.nan] = 0
    return out

# def elnsum(x, y):
#     out = x
#     out[x==0] = eln(y[x==0])
#     out[y==0] = eln(x[y==0])
#     out[(x!=0) & (y!=0)] = eln((x+y)[(x!=0) & (y!=0)])
#     return out

def elnsum(elnx, elny):
    if elnx == torch.nan:
        return elny
    elif elny == torch.nan:
        return elnx
    if elnx > elny:
        return elnx + eln(1+torch.exp(elny-elnx))
    else:
        return elny + eln(1+torch.exp(elnx-elny))

def elnproduct(elnx, elny):
    return elnx + elny # this automatically handles nan values correctly

test = torch.zeros(10)
assert not torch.equal(test, eln(test))
test = eln(test)
assert not torch.equal(test, eexp(test))


def forward_pass(prior, emission_probs, cur_A):
    assert emission_probs.shape[1:] == prior.shape, f"Shape mismatch: {emission_probs.shape = }, {prior.shape = }"
    assert cur_A.shape == (len(prior), len(prior)), f"Shape mismatch: {cur_A.shape = }, {prior.shape = }"

    eln_A = eln(cur_A)
    eln_prior = eln(prior)
    eln_emission = eln(emission_probs)

    eln_alphas = torch.empty_like(emission_probs)
    eln_alphas[0,:] = elnproduct(eln_prior, eln_emission[0,:])

    for t in range(1, len(emission_probs)):
        print(f'{t = }')
        for j in range(len(prior)):
            # print(f'{j = }')
            logalpha = torch.nan
            temp = elnproduct(eln_alphas[t-1], eln_A[:,j])
            # cheat for now
            temp = torch.exp(temp).sum().log()
            # for i in range(len(prior)):
            #     # print(f'{i = }')
            #     logalpha = elnsum(logalpha, temp[i])
            # eln_alphas[t,j] = elnproduct(logalpha, eln_emission[t,j])
            eln_alphas[t,j] = elnproduct(temp, eln_emission[t,j])
    print(f"{eln_alphas = }")
    return eln_alphas

def backward_pass(emission_probs, cur_A):
    eln_betas = torch.zeros_like(emission_probs)
    eln_A = eln(cur_A)

    for t in range(len(emission_probs)-1,-1,-1): #TODO # maybe smth going wrong with the time indexing here
        print(f"{t = }")
        for i in range(len(prior)):
            logbeta = torch.nan
            # cheat for now 
            logbeta = torch.exp(elnproduct(eln_A[i,:], elnproduct(emission_probs[t], eln_betas[t]))).sum().log()
            # for j in range(len(prior)):
            #     print(f'{j = }')
            #     logbeta = elnsum(logbeta, elnproduct(eln_A[i,j], elnproduct(emission_probs[j], eln_betas[-1][j])))
            eln_betas[t-1] = logbeta
    print(f"{eln_betas = }")
    return eln_betas

def compute_eln_gammas(eln_alphas, eln_betas):
    eln_gammas = elnproduct(eln_alphas, eln_betas)
    print(f"{eln_gammas = }")
    normalizer = torch.exp(eln_gammas).sum(dim=1)
    print(f"{normalizer = }")
    normalizer = normalizer.log()
    print(f"{normalizer = }")

    eln_gammas = eln_gammas - normalizer[:,None]
    # for t in range(eln_gammas.shape[0]):
    #     normalizer = torch.nan
    #     for i in range(eln_gammas.shape[1]):
    #         normalizer = elnsum(normalizer, eln_gammas[t,i])
    #     for i in range(eln_gammas.shape[1]):
    #         eln_gammas[t,i] = elnproduct(eln_gammas[t,i], -normalizer)
    return eln_gammas

def compute_eln_xis(eln_alphas, eln_betas, emission_probs, cur_A):
    eln_emission = eln(emission_probs)
    eln_xis = torch.empty(*eln_alphas.shape, eln_alphas.shape[1], device=cur_A.device)
    eln_A = eln(cur_A)
    eln_xis = eln_alphas[:,:,None] + eln_A[None,:,:] + eln_emission[:,None,:] + eln_betas[:,None,:]
    normalizer = torch.exp(eln_xis).sum(dim=[1,2]).log()
    eln_xis = eln_xis - normalizer[:,None,None]
    
    # for t in range(eln_alphas.shape[0]):
    #     print(f"{t = }")
    #     # normalizer = torch.nan
    #     eln_xis[t] = elnproduct(eln_alphas[t], elnproduct(eln_A, elnproduct(eln_emission[t], eln_betas[t])))
    #     # cheat for now
    #     # for i in range(cur_A.shape[1]):
    #     #     for j in range(cur_A.shape[1]):
    #     #         normalizer = elnsum(normalizer, eln_xis[t,i,j])
    #     eln_xis[t] = elnproduct(eln_xis[t], -normalizer)
        # for i in range(eln_alphas.shape[1]):
        #     for j in range(eln_alphas.shape[1]):
        #         eln_xis[t,i,j] = elnproduct(eln_xis[t,i,j], -normalizer)

    return eln_xis

def update_A(eln_gammas, eln_xis):
    A = torch.empty((eln_gammas.shape[1], eln_gammas.shape[1]))
    
    #cheat for now
    # print(f"{eln_xis = }")
    # print(f"{eln_gammas = }")
    numerator = torch.exp(eln_xis).sum(dim=0).log()
    denominator = torch.exp(eln_gammas).sum(dim=0).log()
    print(f"{numerator - denominator = }")
    A = eexp(elnproduct(numerator, -denominator))
    print(f'internal A = {A}')
    # for i in range(eln_gammas.shape[1]):
    #     for j in range(eln_gammas.shape[1]):
    #         numerator = torch.nan
    #         denominator = torch.nan
    #         for t in range(eln_gammas.shape[0]):
    #             numerator = elnsum(numerator, eln_xis[t,i,j])
    #             denominator = elnsum(denominator, eln_gammas[t,i])
    #         A[i,j] = eexp(elnproduct(numerator, -denominator))
    return A


EPS = torch.finfo(torch.float64).eps

device = 'cuda' if torch.cuda.is_available() else 'cpu'

CONSTANT_ENV = True

data_file = "/home/aric/Desktop/Projects/Master Thesis/ppo_const_env_experience.npz" if CONSTANT_ENV else "/home/aric/Desktop/Projects/Master Thesis/ppo_changing_env_experience.npz"
checkpoint_path = '/home/aric/Desktop/Projects/Master Thesis/Crossing-dVAE/2fxh6iq4/checkpoints/last.ckpt' if CONSTANT_ENV else '/home/aric/Desktop/Projects/Master Thesis/Crossing-dVAE/12994q6g/checkpoints/last.ckpt' #TODO
results_file = "/home/aric/Desktop/Projects/Master Thesis/A_const.npz" if CONSTANT_ENV else "/home/aric/Desktop/Projects/Master Thesis/A_changing.npz"
repr_model = dVAE.load_from_checkpoint(checkpoint_path)
repr_model.to(device)

# init A to uniform
A = torch.ones((1024,1024), device=device) / 1024

# set prior over latent #TODO
prior = torch.ones(1024, device=device) / 1024

# load data
data = np.load(data_file)
obs = data['obs']
done = data['done']

# split trajectories
stop = np.argwhere(done == 1)
num_trajectories = len(stop)

# for every latent state compute the emission probabilities -> can be reused for every trajectory
latents = torch.arange(1024, device=device)
latents = torch.nn.functional.one_hot(latents).float()
emission_probs = repr_model.quantize_decode(latents)
emission_probs = einops.rearrange(emission_probs, 'b c h w -> b c (h w)').reshape(1024,4,49)

prev = 0
# for traj in tqdm(range(num_trajectories)):
for traj in tqdm(range(5)):
    # traj_obs = obs[prev:stop[traj,0]]
    # prev = stop[traj,0]
    traj_obs = obs[0:stop[0,0]]
    traj_obs = traj_obs.reshape(traj_obs.shape[0],-1)

    print(f"{traj_obs.shape = }")
    print(f"{traj_obs[0][0] = }")
    print(f"{emission_probs[0,:,0] = }")
    traj_emission_probs = torch.stack([emission_probs[:, traj_obs[i], torch.arange(emission_probs.shape[-1])] for i in range(len(traj_obs))], dim=-1)
    print(traj_emission_probs.shape)
    traj_emission_probs = torch.exp(torch.log(traj_emission_probs).sum(dim=1)) # is better for stability
    print(f"{traj_emission_probs = }")
    # break
    traj_emission_probs = einops.rearrange(traj_emission_probs, 'd t -> t d')
    print(traj_emission_probs[11])
    eln_alphas = forward_pass(prior, traj_emission_probs, A)
    eln_betas = backward_pass(traj_emission_probs, A)
    eln_gammas = compute_eln_gammas(eln_alphas, eln_betas)
    print(f'{eln_gammas = }')
    eln_xis = compute_eln_xis(eln_alphas, eln_betas, traj_emission_probs, A)
    print(f'{eln_xis = }')
    print('update A')
    A = update_A(eln_gammas, eln_xis)
    # print(A)
    # A = A + EPS
    # print(A)
    # A /= A.sum(dim=1)[:,None]
    print(A)
    print(f"{A.sum(dim=1) = }")
    print(f"{A.sum(dim=0) = }")
    print(A[A==0])
    assert not np.isnan(A.cpu().numpy().sum()), "NaN 1"
# #TODO
A = A.detach().cpu().numpy()
print(A)
print(A.max(axis=1))
print(A.max(axis=0))
# # -> 'problem' is that A is almost perfectly deterministic, except for ~2700 entries which are non-0 and non-1

# # this might only be a result of having the determinstic, fixed environment though.
# print(len(A[(A != 0) & (A != 1)]))
# assert not np.isnan(A.sum()), "NaN values in A encountered"

# np.savez_compressed(results_file, A=A)