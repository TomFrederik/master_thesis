import argparse
import math
import os
import sys

import einops
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
import wandb
from torchvision.utils import make_grid

sys.path.insert(0, '../')
from hmm.datasets import construct_toy_train_val_data, construct_pong_train_val_data
from dreamer_world_model import DreamerWorldModel
from parsers import create_train_parser

class ReconstructionCallback(pl.Callback):
    def __init__(self, dataset, every_n_batches=100) -> None:
        super().__init__()
        self.every_n_batches = every_n_batches
        
        obs, actions, vp, nonterms, dropped, player_pos = dataset.get_no_drop(0)
        self.obs = torch.from_numpy(obs[:,None])
        self.actions = torch.from_numpy(actions[:,None])
        if player_pos is not None:
            self.player_pos = torch.from_numpy(player_pos[:,None])
        else:
            self.player_pos = None
        self.nonterms = torch.from_numpy(nonterms[:,None,...,None])
        self.value_prefixes = torch.from_numpy(vp[:,None])
        self.dropped = torch.from_numpy(dropped[:,None])
        self.mu = dataset.mu
        self.sigma = dataset.sigma
        
    
    def on_batch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        if (pl_module.global_step+1) % self.every_n_batches == 0:
            self.visualize_recon(trainer, pl_module)
    
    def visualize_recon(self, trainer, pl_module):
        if self.obs.device != pl_module.device:
            self.obs = self.obs.to(pl_module.device)
            self.actions = self.actions.to(pl_module.device)
            self.actions = torch.nn.functional.one_hot(self.actions, pl_module.action_size)
            self.dropped = self.dropped.to(pl_module.device)
            if self.player_pos is not None:
                self.player_pos = self.player_pos.to(pl_module.device)
            self.nonterms = self.nonterms.to(pl_module.device)
            self.value_prefixes = self.value_prefixes.to(pl_module.device)

        *_, obs_hat = pl_module.representation_loss(self.obs, self.actions, self.value_prefixes, self.nonterms, self.dropped, batch_size=1)
        # obs_hat = einops.rearrange(obs_hat, '... (h w) -> ... h w', h=self.obs.shape[-2], w=self.obs.shape[-1])
        num_views = self.obs.shape[2]
        images = torch.stack(
            [(self.obs[:,0,i].to('cpu')*self.sigma)+self.mu for i in range(num_views)]\
            + [(obs_hat[:,0,i].to('cpu')*self.sigma)+self.mu for i in range(num_views)],
            dim = 1
        ).reshape(2*num_views*obs_hat.shape[0], 1, obs_hat.shape[-2], obs_hat.shape[-1])

        # log images
        pl_module.logger.experiment.log({'Reconstruction': wandb.Image(make_grid(images, nrow=2*num_views))})

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
        obs, actions, vp, nonterms, dropped, playerpos = dataset.get_no_drop(0)
        self.obs = torch.from_numpy(obs)
        self.actions = torch.from_numpy(actions[None])
        self.mu = dataset.mu
        self.sigma = dataset.sigma

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
            # convert actions to one-hot vectors
            self.actions = torch.nn.functional.one_hot(self.actions, pl_module.action_size)
            self.actions = einops.rearrange(self.actions, 'b t ... -> t b ...')
            
        num_views = self.obs.shape[1]
        obs_mean = pl_module.extrapolate_from_init_obs(self.obs[None,0], self.actions) # is just mean
        # obs_mean = einops.rearrange(obs_mean, 'b c (h w) -> b c h w', h=7, w=7)
        images = torch.stack(
                [(self.obs[:,i].to('cpu')*self.sigma)+self.mu for i in range(num_views)]\
                + [(obs_mean[:,i].to('cpu')*self.sigma)+self.mu for i in range(num_views)],
                dim = 1
            ).reshape(2*num_views*obs_mean.shape[0], 1, self.obs.shape[2], self.obs.shape[3])

        # log images
        pl_module.logger.experiment.log({'Extrapolation': wandb.Image(make_grid(images, nrow=2*num_views))})

def main(
    num_views,
    view_null_value,
    drop_null_value,
    percentage,
    dropout,
    kl_balancing_coeff,
    kl_scaling,
    discount_factor,
    learning_rate,
    weight_decay,
    batch_size,
    num_epochs,
    num_workers,
    seed,
    test_only_dropout,
    max_len,
    gradient_clip_val,
    depth,
    kernel_size,
    codebook_size,
    num_variables,
    wandb_group,
    wandb_id,
    obs_scale,
):
    
    
    test_only_dropout = test_only_dropout == 'yes'
    
    
    data_cls = construct_toy_train_val_data
    data_kwargs = dict(
        num_views=num_views,
        view_null_value=view_null_value,
        drop_null_value=drop_null_value,
        percentage=percentage,
        dropout=dropout,
        test_only_dropout=test_only_dropout,
        train_val_split=0.9,
        batch_size=batch_size,
        max_len=max_len,
        scale=obs_scale,
        get_player_pos=False,
    )
    
    pl.seed_everything(seed)
    
    # get data path
    file_name = f"ppo_all_env_experience.npz"
    data_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(data_path, file_name)
    
    # init dataset and dataloader
    train_val_split = 0.9
    train_data, val_data = data_cls(data_path, **data_kwargs)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    # settings for toy env
    num_actions = 4
    
    rssm_type: str = 'discrete'
    embedding_size: int = 128
    rssm_node_size: int = 128
    rssm_info = {'deter_size':100, 'class_size':num_variables, 'category_size':codebook_size, 'min_std':0.1}

    obs_encoder = {'layers':3, 'dist': None, 'activation':torch.nn.ELU, 'kernel':kernel_size, 'depth':depth} # , 'node_size':100
    obs_decoder = {'layers':3, 'dist': None, 'activation':torch.nn.ELU, 'kernel':kernel_size, 'depth':depth} # , 'node_size':100
    worldmodel_config = dict(
        num_views=num_views,
        action_size = num_actions,
        kl_info = {'use_kl_balance':True, 'kl_balance_scale':kl_balancing_coeff, 'use_free_nats':False},
        lr = learning_rate,
        weight_decay = weight_decay,
        batch_size = batch_size,
        loss_scale = {'kl':kl_scaling},
        rssm_node_size = rssm_node_size,
        rssm_type = rssm_type,
        rssm_info = rssm_info,
        embedding_size = embedding_size,
        obs_shape = (num_views,7,7),
        num_variables = num_variables,
        depth = depth,
        kernel_size = kernel_size,
        obs_encoder = obs_encoder,
        obs_decoder = obs_decoder,
        reward_config = dict(
            layers = 2,
            activation = torch.nn.ReLU,
            node_size = 128,
            dist = None,
        ),
        view_masks = train_data.mvwrapper.view_masks,
    )
    
    
    model = DreamerWorldModel(
        argparse.Namespace(**worldmodel_config),
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    )
    
    # set up wandb
    wandb_config = dict(
        seed=seed,
        kl_balancing_coeff=kl_balancing_coeff,
        kl_scaling=kl_scaling,
        learning_rate=learning_rate,
        num_views=num_views,
        percentages=[percentage]*num_views,
        dropout=dropout,
        test_only_dropout=test_only_dropout,
        weight_decay=weight_decay,
        gradient_clip_val=gradient_clip_val,
        max_len=max_len,
    )
    wandb_kwargs = dict(project="MT-ToyTask-Dreamer", config=wandb_config, group=wandb_group, id=wandb_id)
    logger = WandbLogger(**wandb_kwargs)
    
    # callbacks
    callbacks = []
    callbacks.append(pl.callbacks.ModelCheckpoint(save_top_k=1, verbose=True))
    callbacks.append(pl.callbacks.TQDMProgressBar(refresh_rate=1))
    callbacks.append(ExtrapolateCallback(dataset=val_data, every_n_batches=100))
    callbacks.append(ReconstructionCallback(dataset=val_data, every_n_batches=100))
    
    # set up lightning trainer
    trainer = pl.Trainer(
        gpus=torch.cuda.device_count(),
        max_epochs=num_epochs,
        callbacks=callbacks,
        logger=logger,
        gradient_clip_val=gradient_clip_val,
    )
    
    # start training
    trainer.fit(model, train_loader, val_loader)
    
    # finish logging
    logger.experiment.finish()

if __name__ == '__main__':
    parser = create_train_parser()
    
    args = parser.parse_args()
    
    main(**vars(args))
    