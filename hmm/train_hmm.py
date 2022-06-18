import argparse
import logging
import math
import os
from collections import namedtuple

import einops
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader
from torchvision.utils import make_grid

import wandb
from datasets import TransitionData, construct_train_val_data
from models import LightningNet, sum_factored_logits
from sparsemax_k import sparsemax_k

from callbacks import ExtrapolateCallback, ReconstructionCallback

RewardSupport = namedtuple("RewardSupport", ["min", "max", "size"])

        
def main(
    num_seeds,
    num_views,
    null_value,
    percentage,
    dropout,
    kl_balancing_coeff,
    kl_scaling,
    l_unroll,
    discount_factor,
    num_variables,
    codebook_size,
    embedding_dim,
    learning_rate,
    weight_decay,
    batch_size,
    num_epochs,
    num_workers,
    accumulate_grad_batches,
    # entropy_scale,
    gradient_clip_val,
    mlp_repr,
    seed,
    detect_anomaly,
    test_only_dropout,
    max_datapoints,
    track_grad_norm,
    disable_recon_loss,
    sparsemax,
    sparsemax_k,
    disable_vp,
    action_layer_dims,
    max_len,
    traj_max_len,
    vp_batchnorm,
    force_uniform_prior,
    prior_noise_scale,
    obs_scale,
):
    # parse 'boolean' arguments (this needs to be done to be able to give them to the sweeper.. cringe)
    sparsemax = sparsemax == 'yes'
    test_only_dropout = test_only_dropout == 'yes'
    disable_vp = disable_vp == 'yes'
    disable_recon_loss = disable_recon_loss == 'yes'
    mlp_repr = mlp_repr == 'yes'
    vp_batchnorm = vp_batchnorm == 'yes'
    force_uniform_prior = force_uniform_prior == 'yes'
    
    if codebook_size != 2:
        raise NotImplementedError("Codebook size other than 2 is not supported")

        
    percentages = [percentage] * num_views
    
    pl.seed_everything(seed)
    
    # get data path
    suffix = 'all' if num_seeds is None else str(num_seeds)
    file_name = f"ppo_{suffix}_env_experience.npz"
    data_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(data_path, file_name)
    
    # init dataset and dataloader
    data_kwargs = dict(
        multiview=num_views > 1,
        null_value=null_value,
        percentages=percentages,
        dropout=dropout,
        max_datapoints=max_datapoints,
        test_only_dropout=test_only_dropout,
        train_val_split=0.9,
        batch_size=batch_size,
        max_len=max_len,
        traj_max_len=traj_max_len,
        scale=obs_scale,
    )
    
    
    train_data, val_data = construct_train_val_data(data_path, **data_kwargs)
            
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=num_workers)#, collate_fn=TransitionData.collate_fn)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, num_workers=num_workers)#, collate_fn=TransitionData.collate_fn)
    
    # settings for toy env
    num_actions = 4
    num_input_channels = len(percentages)

    # emission settings    
    emission_kwargs = dict(
        num_input_channels=num_input_channels,
        codebook_size=codebook_size,
        embedding_dim=embedding_dim,
        num_variables=num_variables,
        mlp=mlp_repr,
        sparse=sparsemax,
        scale=obs_scale,
    )
    
    num_values = 1
    mlp_hidden_dims = [512, 512, 256, 128]
    vp_kwargs = dict(
        num_values=num_values,
        mlp_hidden_dims=mlp_hidden_dims,
        vp_batchnorm=vp_batchnorm,
    )
    
    model = LightningNet(
        codebook_size ** num_variables,
        num_actions,
        vp_kwargs,
        emission_kwargs,
        kl_balancing_coeff,
        l_unroll,
        discount_factor,
        learning_rate,
        weight_decay,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        reward_support=RewardSupport(0,1,num_values),
        disable_recon_loss=disable_recon_loss,
        sparsemax=sparsemax,
        sparsemax_k=sparsemax_k,
        disable_vp=disable_vp,
        action_layer_dims=action_layer_dims,
        kl_scaling=kl_scaling,
        force_uniform_prior=force_uniform_prior,
        prior_noise_scale=prior_noise_scale,
    )
    
    # set up wandb
    wandb_config = dict(
        seed=seed,
        codebook_size=codebook_size,
        num_variables=num_variables,
        num_seeds=num_seeds,
        embedding_dim=embedding_dim,
        kl_balancing_coeff=kl_balancing_coeff,
        l_unroll=l_unroll,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        gradient_clip_val=gradient_clip_val,
        mlp_repr=mlp_repr,
        num_views=num_views,
        percentages=percentages,
        dropout=dropout,
        test_only_dropout=test_only_dropout,
        max_datapoints=max_datapoints,
        disable_recon_loss=disable_recon_loss,
        sparsemax=sparsemax,
        sparsemax_k=sparsemax_k,
        disable_vp=disable_vp,
        action_layer_dims=action_layer_dims,
        max_len=max_len,
        kl_scaling=kl_scaling,
        vp_batchnorm=vp_batchnorm,
        force_uniform_prior=force_uniform_prior,
        prior_noise_scale=prior_noise_scale,
        obs_scale=obs_scale,
    )
    wandb_kwargs = dict(project="MT-ToyTask-Ours", config=wandb_config)
    logger = WandbLogger(**wandb_kwargs)
    
    # callbacks
    callbacks = []
    callbacks.append(pl.callbacks.ModelCheckpoint(save_top_k=1, verbose=True))
    callbacks.append(pl.callbacks.TQDMProgressBar(refresh_rate=1))
    callbacks.append(ExtrapolateCallback(dataset=val_data, every_n_batches=50))
    callbacks.append(ReconstructionCallback(dataset=train_data, every_n_batches=50))
    
    # set up lightning trainer
    trainer = pl.Trainer(
        gpus=torch.cuda.device_count(),
        max_epochs=num_epochs,
        callbacks=callbacks,
        logger=logger,
        accumulate_grad_batches=accumulate_grad_batches,
        gradient_clip_val=gradient_clip_val,
        detect_anomaly=detect_anomaly,
        track_grad_norm=track_grad_norm,
        log_every_n_steps=1,
        num_sanity_val_steps=1,
    )
    # start training
    trainer.fit(model, train_loader, val_loader)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    # env settings
    parser.add_argument('--num_seeds', type=int, default=None)
    parser.add_argument('--null_value', type=int, default=1)
    parser.add_argument('--num_views', type=int, default=1)
    parser.add_argument('--percentage', type=float, default=1)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--test_only_dropout', default='no', type=str, choices=['yes', 'no'])
    parser.add_argument('--max_datapoints', type=int, default=None)
    parser.add_argument('--obs_scale', type=int, default=1)
    
    ## model args
    parser.add_argument('--kl_balancing_coeff', type=float, default=0.8)
    parser.add_argument('--kl_scaling', type=float, default=0.1)
    parser.add_argument('--l_unroll', type=int, default=1)
    parser.add_argument('--discount_factor', type=float, default=0.99)
    parser.add_argument('--num_variables', type=int, default=10)
    parser.add_argument('--codebook_size', type=int, default=2)
    parser.add_argument('--embedding_dim', type=int, default=128)
    parser.add_argument('--mlp_repr', default='no', type=str, choices=['yes', 'no'])
    parser.add_argument('--disable_recon_loss', default='no', type=str, choices=['yes', 'no'])
    parser.add_argument('--sparsemax', default='no', type=str, choices=['yes', 'no'])
    parser.add_argument('--disable_vp', default='no', type=str, choices=['yes', 'no'])
    parser.add_argument('--sparsemax_k', type=int, default=30)
    parser.add_argument('--action_layer_dims', type=int, nargs='*', default=None)
    parser.add_argument('--vp_batchnorm', type=str, choices=['yes', 'no'], default='no')
    parser.add_argument('--force_uniform_prior', type=str, choices=['yes', 'no'], default='no')
    parser.add_argument('--prior_noise_scale', type=float, default=0.0)
    
    # training args
    parser.add_argument('--learning_rate', type=float, default=0.0002)
    parser.add_argument('--weight_decay', type=float, default=0.000001)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--accumulate_grad_batches', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--num_epochs', type=float, default=10)
    parser.add_argument('--gradient_clip_val', type=float, default=0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--detect_anomaly', action='store_true')
    parser.add_argument('--track_grad_norm', type=int, default=-1)
    parser.add_argument('--traj_max_len', type=int, default=20, help='Max length of an episode. Longer episodes will be discarded')
    parser.add_argument('--max_len', type=int, default=10, help='Max length of an episode for batching purposes. Rest will be padded.')
    
    args = parser.parse_args()
    
    main(**vars(args))
    
