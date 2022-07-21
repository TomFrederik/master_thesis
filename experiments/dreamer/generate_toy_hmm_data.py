import einops
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

from dvae import dVAE


# ###############
# ###############

# # transition matrix
# true_A = np.array([
#     [0.25, 0.75],
#     [0.75, 0.25],
# ])

# # observation matrix
# B = np.eye(2)

# obs = [np.array([0,1,0,0,1,0,1,1])]
# print(f"{len(obs) = }")
# print(obs)
# ###############
# ###############


# device = 'cuda' if torch.cuda.is_available() else 'cpu'

# # init A to uniform
# # A = torch.ones((3,3), device=device, dtype=torch.float64) / 3
# A = torch.rand((2,2), device=device, dtype=torch.float64)
# A /= A.sum(dim=1)[:, None]
# A = torch.from_numpy(true_A).to(device)
# accum_A = torch.zeros_like(A)
# print(f'initial A = {A}')

# # set prior over latent #TODO
# prior = torch.ones(2, device=device) / 2
    
# for traj in tqdm(range(len(obs))):
#     traj_obs = obs[traj]
#     traj_emission_probs = torch.from_numpy(B[:, traj_obs]).to(A.device)
#     print(f"{traj_emission_probs.shape = }")
#     T = traj_emission_probs.shape[1]
#     print(f"{traj_emission_probs = }")

#     # forward
#     alpha_0 = prior.clone() * traj_emission_probs[:,0]
#     print(f"{alpha_0 = }")
#     constants = [alpha_0.sum()]
#     alphas = [alpha_0/constants[0]]
#     for t in range(1,len(traj_obs)):
#         new_alpha = (alphas[-1] @ A) * traj_emission_probs[:,t]
#         constants.append(new_alpha.sum())
#         new_alpha /= constants[-1]
#         # print(f"{traj_emission_probs[:,t] = }")
#         # print(f"{new_alpha = }")
#         alphas.append(new_alpha)
#     alphas = torch.stack(alphas, dim=-1)
#     assert alphas.shape[1] == T, f"{alphas.shape = }"
    
#     # backward
#     betas = [torch.ones_like(prior)]
#     for t in range(1, len(traj_obs)):
#         # print(f"{t = }")
#         new_beta = A @ (betas[-1] * traj_emission_probs[:,-t])
#         # print(f"{traj_emission_probs[:,-t] = }")
#         new_beta = new_beta / constants[-t]
#         # print(f'{new_beta = }')
#         betas.append(new_beta)
#     # break
#     betas = torch.stack(betas[::-1], dim=-1)
#     assert betas.shape[1] == T, f"{betas.shape = }"
#     # print(betas.shape)
#     constants = torch.stack(constants, dim=-1)

#     # update A
#     print(f"{constants = }")
#     # print(f"{traj_emission_probs = }")1
#     print(f"{alphas = }")
#     print(f"{betas = }")
#     # print(f"{torch.einsum('it, jt -> ijt', alphas[:,:-1], betas[:,1:]) = }")
#     gammas = alphas * betas
#     print(f"{gammas = }")
#     print(f"{gammas.sum(dim=0) = }")
#     # xi_ijt =  A[:,:,None] * traj_emission_probs[None,:,1:] * torch.einsum('it, jt -> ijt', alphas[:,:-1], betas[:,1:])
#     xi_ijt = torch.empty((2,2,len(traj_obs)-1), device=device, dtype=torch.float64)
#     for i in range(2):
#         for j in range(2):
#             for t in range(len(traj_obs)-1):
#                 xi_ijt[i,j,t] = 1/constants[t+1] * A[i,j] * traj_emission_probs[j,t+1] * alphas[i,t] * betas[j,t+1]
#     # print(xi_ijt[0,1,0])
#     # test = torch.empty(2,2,2)
#     # xi_ijt =  1/constants[None, None, 1:] * A[...,None] * traj_emission_probs[None,:,1:] * torch.einsum('it, jt -> ijt', alphas[:,:-1], betas[:,1:])
#     # print(f"{xi_ijt = }")
#     # print(f"{xi_ijt.sum(dim=0) = }")
#     # print(f"{xi_ijt.sum(dim=1) = }")
#     # print(f"{xi_ijt.sum(dim=[0,1]) = }")
#     # print(f"{xi_ijt = }")
#     denominator = xi_ijt.sum(dim=2)
#     numerator = denominator.sum(dim=1)
#     accum_A += (denominator / numerator[:,None])
#     assert not np.isnan(accum_A.cpu().numpy().sum()), accum_A
#     # print(f"{A = }")
#     # print(f"{A.sum(dim=1) = }")
#     # print(f"{A.sum(dim=0) = }")
#     # print(A[A==0])
#     print(accum_A/(traj+1))

#     assert not np.isnan(A.cpu().numpy().sum()), "NaN 1"
# # #TODO
# estimate_A = accum_A / len(obs)
# print(accum_A)
# print(true_A)
# print(estimate_A)



###############
###############

# transition matrix
true_A = np.array([
    [0.1, 0.3, 0.6],
    [0.5, 0.01, 0.49],
    [0.2, 0.7, 0.1],
])
print(f'{true_A = }')


# observation matrix
B = np.eye(3) + 0.01
B /= B.sum(axis=1)[:,None]
print(f'{B = }')

# generate observations
num_trajectories = 100
traj_length = 100

obs = []
for n in range(num_trajectories):
    idx = np.random.choice(3)
    p = np.zeros(3)
    p[idx] = 1

    obs.append([np.random.choice(3, p = p @ B)])
    for t in range(traj_length):
        p = p @ true_A
        obs[-1].append(np.random.choice(3, p=p@B))

print(f"{len(obs) = }")
print(obs)
###############
###############


device = 'cuda' if torch.cuda.is_available() else 'cpu'

# init A to uniform
A = torch.ones((3,3), device=device, dtype=torch.float64) / 3
print(f'initial A = {A}')

# set prior over latent #TODO
prior = torch.ones(3, device=device) / 3

for k in range(10):    
    for traj in tqdm(range(len(obs))):
        traj_obs = obs[traj]
        traj_emission_probs = torch.from_numpy(B[:, traj_obs]).to(A.device)
        T = traj_emission_probs.shape[1]

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
        xi_ijt =  1/constants[None, None, 1:] * A[...,None] * traj_emission_probs[None,:,1:] * torch.einsum('it, jt -> ijt', alphas[:,:-1], betas[:,1:])
        denominator = xi_ijt.sum(dim=2)
        numerator = denominator.sum(dim=1)
        A = (denominator / numerator[:,None])

        assert not np.isnan(A.cpu().numpy().sum()), "NaN 1"
# #TODO
estimate_A = A
print(true_A)
print(estimate_A)

