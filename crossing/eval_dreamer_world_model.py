import os
import sys
from dreamer_world_model import DreamerWorldModel
sys.path.insert(0, '../')
from hmm.datasets import SingleTrajToyData
from dreamerv2.utils.rssm import RSSMDiscState
import torch
import einops
import pytorch_lightning as pl
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
seed = 42
pl.seed_everything(seed)

## single
# changing
path = "/home/aric/Desktop/Projects/Master Thesis/crossing/MT-ToyTask-Dreamer/1u4t9w76/checkpoints/epoch=0-step=8999.ckpt"
# constant
path = "/home/aric/Desktop/Projects/Master Thesis/crossing/MT-ToyTask-Dreamer/1j630ppa/checkpoints/epoch=1-step=1799.ckpt"

## 50% drop
# changing
# path = "/home/aric/Desktop/Projects/Master Thesis/crossing/MT-ToyTask-Dreamer/5jsch67d/checkpoints/epoch=0-step=8999.ckpt"
# constant


## 10% drop
# changing
# path = "/home/aric/Desktop/Projects/Master Thesis/crossing/MT-ToyTask-Dreamer/3ts9knja/checkpoints/epoch=0-step=8999.ckpt"
# constant
path = "/home/aric/Desktop/Projects/Master Thesis/crossing/MT-ToyTask-Dreamer/3vqsj5za/checkpoints/epoch=0-step=899.ckpt"

## 0% drop
# changing
# constant
# path = "/home/aric/Desktop/Projects/Master Thesis/crossing/MT-ToyTask-Dreamer/138pup81/checkpoints/epoch=1-step=1799.ckpt"



model = DreamerWorldModel.load_from_checkpoint(path).cuda()

const = True
const = "const" if const else "changing"
file_name = f"ppo_{const}_env_experience.npz"
data_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_path = os.path.join(data_path, file_name)

percentages = [0.5,0.5]
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


# convert actions to one-hot vectors
actions = torch.nn.functional.one_hot(actions, model.action_size)

nonterms = (1-terms).unsqueeze(-1)

obs = einops.rearrange(obs, 'b t ... -> t b ...')
actions = einops.rearrange(actions, 'b t ... -> t b ...')
nonterms = einops.rearrange(nonterms, 'b t ... -> t b ...')

print(obs.shape)
obs_mean = model.extrapolate_from_init_obs(obs[0], actions) # is just mean
obs_mean = einops.rearrange(obs_mean, '(t b) (h w) -> t 1 b h w', b=len(percentages), h=7, w=7)

num_views = len(percentages)

images = torch.stack(
        [(obs[1:,:,i].to('cpu')*dataset.sigma)+dataset.mu for i in range(num_views)]\
        + [(obs_mean[:,:,i].to('cpu')*dataset.sigma)+dataset.mu for i in range(num_views)],
        dim = 1
    )[:,:,0]#.reshape(2*num_views*obs_mean.shape[0], *obs.shape[-2:])

images = images.detach().numpy()
fig, axes = plt.subplots(obs_mean.shape[0], 2*num_views, sharex=True, sharey=True, gridspec_kw={'hspace': 0, 'wspace': -.8})
for row in range(obs_mean.shape[0]):
    for col in range(2*num_views):
        axes[row,col].imshow(images[row, col], cmap='gray', vmin=0, vmax=2)
        axes[row,col].axis('off')

# plt.imshow(images, cmap='gray')
plt.show()