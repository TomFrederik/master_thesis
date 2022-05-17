



import os
import sys
from models import LightningNet
sys.path.insert(0, '../')
from hmm.datasets import SingleTrajToyData
from dreamerv2.utils.rssm import RSSMDiscState
import torch
import einops
import pytorch_lightning as pl
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
import numpy as np

seed = 42
pl.seed_everything(seed)

path = "/home/aric/Desktop/Projects/Master Thesis/hmm/MT-ToyTask-Ours/2z68drf9/checkpoints/epoch=9-step=8999.ckpt"



model = LightningNet.load_from_checkpoint(path).cuda()

const = True
const = "const" if const else "changing"
file_name = f"ppo_{const}_env_experience.npz"
data_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_path = os.path.join(data_path, file_name)

percentages = [1]
multiview = len(percentages) > 1

data_kwargs = dict(
    multiview=multiview,
    null_value=1,
    percentages=percentages,
    dropout=0,
)
dataset = SingleTrajToyData(data_path, **data_kwargs)


obs, actions, terms, dropped, player_pos = dataset[0]
obs = torch.from_numpy(obs)[None].cuda()
actions = torch.from_numpy(actions)[None].cuda()
terms = torch.from_numpy(terms)[None].cuda()


nonterms = (1-terms).unsqueeze(-1)

obs = einops.rearrange(obs, 'b t ... -> t b ...')
nonterms = einops.rearrange(nonterms, 'b t ... -> t b ...')


if obs.device.type != 'cuda':
    obs = obs.to('cuda')
    actions = actions.to('cuda')
    dropped = dropped.to('cuda')
# print(.transition_matrices.matrices)

# extrapolations
obs_logits = model.emission(obs)

prior = model.prior(1) # batch size 1
ent = -(prior * prior.log())
ent[prior == 0] = 0
ent = ent.sum()
print(f"entropy: {ent}")
print(prior.argmax())

prior[:,prior.argmax(dim=-1)] = 1
prior[prior != 1] = 0
print(f"{prior.shape = }")


initial_obs = model.emission.decode_only(prior)
# plt.figure()
# plt.imshow((dataset[0][0][0,0].astype(np.float32) * dataset.sigma) + dataset.mu, cmap='gray')
# plt.title('True obs t=0')
# plt.colorbar()
# plt.show()

# plt.figure()
# plt.imshow((initial_obs[0,0].detach().cpu().numpy() * dataset.sigma) + dataset.mu, cmap='gray')
# plt.title('Decoded prior t=0')
# plt.colorbar()
# plt.show()

print(torch.softmax(model.network.transition_matrices.matrices[0,:36,:36], dim=-1).detach().cpu().numpy())
plt.figure()
plt.imshow(torch.softmax(model.network.transition_matrices.matrices[0], dim=-1).detach().cpu().numpy(), cmap='gray', origin='lower')
plt.colorbar()
plt.show()

plt.figure()
plt.imshow(torch.softmax(model.network.transition_matrices.matrices[2], dim=-1).detach().cpu().numpy(), cmap='gray', origin='lower')
plt.colorbar()
plt.show()

print(dropped.shape)
obs_logits = einops.rearrange(obs_logits, '(batch seq) views ... -> batch seq views (...)', batch=1, seq=obs.shape[0]) # flatten over latent states
posterior_0 = prior.log() + (obs_logits[:,0] * (1-torch.from_numpy(dropped).cuda())[:,0,None]).sum(dim=1)
posterior_0 = posterior_0.exp() # sum over views and convert to probs for normalization
posterior_0 = posterior_0 / posterior_0.sum(dim=1, keepdim=True)


print(f"{posterior_0.shape = }")
print(f"{actions.shape = }")
state_belief_prior_sequence = model.network.k_step_extrapolation(posterior_0, actions, actions.shape[1])
state_belief_prior_sequence = torch.cat([posterior_0[:,None], state_belief_prior_sequence], dim=1)
for i in range(state_belief_prior_sequence.shape[1]):
    ent = -(state_belief_prior_sequence[:,i] * state_belief_prior_sequence[:,i].log())
    ent[state_belief_prior_sequence[:,i] == 0] = 0
    ent = ent.sum()
    print(f"entropy: {ent}")
emission_input = einops.rearrange(state_belief_prior_sequence, 'b t ... -> (b t) ...')

obs_hat = model.emission.decode_only(emission_input).to('cpu').float()

# for t in range(10):
#     plt.figure()
#     plt.imshow((obs_hat[t,0].detach().cpu().numpy() * dataset.sigma) + dataset.mu, cmap='gray')
#     plt.title(f'Decoded posterior t = {t}')
#     plt.colorbar()
#     plt.show()


print(state_belief_prior_sequence[:,:,35])