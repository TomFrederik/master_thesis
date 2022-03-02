import argparse
import os

import einops
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
import wandb
from torchvision.utils import make_grid

from datasets import SingleTrajToyData
from models import LightningNet

class ExtrapolateCallback(pl.Callback):
    def __init__(self, dataset=None, save_to_disk=False, every_n_batches=100):
        """
        Inputs:
            batch_size - Number of images to generate
            dataset - Dataset to sample from
            save_to_disk - If True, the samples and image means should be saved to disk as well.
        """
        super().__init__()
        self.every_n_batches = every_n_batches
        self.save_to_disk = save_to_disk
        self.initial_loading = False
        self.obs = torch.from_numpy(dataset[0][0])
        self.actions = torch.from_numpy(dataset[0][1][None])
        self.mu = dataset.dataset.mu
        self.sigma = dataset.dataset.sigma

        self.obs_logits = None

    def on_batch_end(self, trainer, pl_module):
        """
        This function is called after every epoch.
        Call the save_and_sample function every N epochs.
        """
        if (pl_module.global_step+1) % self.every_n_batches == 0:
            self.extrapolate(trainer, pl_module, pl_module.global_step+1)
    
    def on_epoch_end(self, trainer, pl_module):
        """
        This function is called after every epoch.
        Call the save_and_sample function every N epochs.
        """
        self.extrapolate(trainer, pl_module, pl_module.global_step+1)

    @torch.no_grad()
    def extrapolate(self, trainer, pl_module, epoch):
        if self.obs.device != pl_module.device:
            self.obs = self.obs.to(pl_module.device)
            self.actions = self.actions.to(pl_module.device)

        self.obs_logits, latent_dist, latent_loss = pl_module.autoencoder(self.obs)
        self.obs_logits = self.obs_logits[0][None]

        self.prior = pl_module.prior(1) # batch size 1
        self.obs_logits = einops.rearrange(self.obs_logits, 'b ... -> b (...)') # flatten over latent states
        self.posterior_0 = (self.prior.log() + self.obs_logits).exp()
        self.posterior_0 = self.posterior_0 / self.posterior_0.sum(dim=1, keepdim=True).detach()
            
        state_belief_prior_sequence = pl_module.network.k_step_extrapolation(self.posterior_0, self.actions, self.actions.shape[1])
        state_belief_prior_sequence = torch.cat([self.posterior_0[:,None], state_belief_prior_sequence], dim=1)

        vae_input = einops.rearrange(state_belief_prior_sequence, 'b t ... -> (b t) ...')
        
        obs_hat = pl_module.autoencoder.decode_only(vae_input).to('cpu').float()
        images = torch.stack([(self.obs.to('cpu') * self.sigma) + self.mu, (obs_hat * self.sigma) + self.mu], dim=1).reshape((2*obs_hat.shape[0], *obs_hat.shape[1:]))
        
        # log images
        pl_module.logger.experiment.log({'Extrapolation': wandb.Image(make_grid(images, nrow=2))})


def main(
    constant_env,
    normalized_float,
    kl_balancing_coeff,
    l_unroll,
    discount_factor,
    num_variables,
    codebook_size,
    embedding_dim,
    learning_rate,
    batch_size,
    num_epochs,
    num_workers,
    accumulate_grad_batches,
    entropy_scale,
    gradient_clip_val,
):
    
    assert batch_size == 1, "batch_size > 1 not implemented"
    
    # get data path
    const = "const" if constant_env else "changing"
    normalized = "_normalized_float" if normalized_float else ""
    file_name = f"ppo_{const}_env_experience{normalized}.npz"
    data_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(data_path, file_name)
    
    # init dataset and dataloader
    data = SingleTrajToyData(data_path)
    train_val_split = 0.9
    train_data_size = int(train_val_split * len(data))
    val_data_size = len(data) - train_data_size
    train_data, val_data = torch.utils.data.random_split(data, [train_data_size, val_data_size])
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    # settings for toy env
    num_actions = 4
    num_input_channels = 1
    
    # dVAE settings    
    vae_kwargs = dict(
        num_input_channels=num_input_channels,
        codebook_size=codebook_size,
        embedding_dim=embedding_dim,
        num_variables=num_variables,
        entropy_scale=entropy_scale,
    )
        
    
    model = LightningNet(
        codebook_size ** num_variables,
        num_actions,
        vae_kwargs,
        kl_balancing_coeff,
        l_unroll,
        discount_factor,
        learning_rate,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    )
    

    # set up wandb
    wandb_config = dict(
        codebook_size=codebook_size,
        num_variables=num_variables,
        constant_env=constant_env,
        embedding_dim=embedding_dim,
        kl_balancing_coeff=kl_balancing_coeff,
        l_unroll=l_unroll,
        learning_rate=learning_rate,
        entropy_scale=entropy_scale,
        gradient_clip_val=gradient_clip_val,
    )
    wandb_kwargs = dict(project="MT-ToyTask-Ours", config=wandb_config)
    logger = WandbLogger(**wandb_kwargs)
    
    # callbacks
    callbacks = []
    callbacks.append(pl.callbacks.ModelCheckpoint(save_top_k=1, verbose=True))
    callbacks.append(pl.callbacks.TQDMProgressBar(refresh_rate=1))
    callbacks.append(ExtrapolateCallback(dataset=val_data, every_n_batches=10))
    
    # set up lightning trainer
    trainer = pl.Trainer(
        gpus=torch.cuda.device_count(),
        max_epochs=num_epochs,
        callbacks=callbacks,
        logger=logger,
        accumulate_grad_batches=accumulate_grad_batches,
        gradient_clip_val=gradient_clip_val,
    )
    
    # start training
    trainer.fit(model, train_loader, val_loader)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    # env settings
    parser.add_argument('--constant_env', action='store_true')
    parser.add_argument('--normalized_float', action='store_true')
    
    ## model args
    parser.add_argument('--kl_balancing_coeff', type=float, default=0.8)
    parser.add_argument('--l_unroll', type=int, default=5)
    parser.add_argument('--discount_factor', type=float, default=0.99)
    parser.add_argument('--num_variables', type=int, default=2)
    parser.add_argument('--codebook_size', type=int, default=10)
    parser.add_argument('--embedding_dim', type=int, default=32)
    parser.add_argument('--entropy_scale', type=float, default=1)
    
    # training args
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--accumulate_grad_batches', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--num_epochs', type=int, default=10)
    parser.add_argument('--gradient_clip_val', type=float, default=0)
    
    
    args = parser.parse_args()
    
    main(**vars(args))
    