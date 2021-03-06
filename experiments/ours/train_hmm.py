import argparse
import os
from collections import namedtuple
import sys
sys.path.append('../../')

import pytorch_lightning as pl
import torch
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader

from src.common.datasets import construct_toy_train_val_data, construct_pong_train_val_data
from src.ours.callbacks import ExtrapolateCallback, ReconstructionCallback
from src.ours.models import LightningNet
from src.ours.parsers import create_train_parser

RewardSupport = namedtuple("RewardSupport", ["min", "max", "size"])

        
def main(
    num_views,
    view_null_value,
    drop_null_value,
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
    vp_layer_dims,
    accumulate_grad_batches,
    # entropy_scale,
    gradient_clip_val,
    seed,
    detect_anomaly,
    test_only_dropout,
    track_grad_norm,
    disable_recon_loss,
    sparsemax,
    sparsemax_k,
    disable_vp,
    action_layer_dims,
    max_len,
    vp_batchnorm,
    force_uniform_prior,
    prior_noise_scale,
    obs_scale,
    kernel_size,
    depth,
    wandb_group,
    wandb_id,
    env_name,
    get_player_pos,
    nonzero_thresh,
):
    
    # parse 'boolean' arguments (this needs to be done to be able to give them to the sweeper.. cringe)
    sparsemax = sparsemax == 'yes'
    test_only_dropout = test_only_dropout == 'yes'
    disable_vp = disable_vp == 'yes'
    disable_recon_loss = disable_recon_loss == 'yes'
    vp_batchnorm = vp_batchnorm == 'yes'
    force_uniform_prior = force_uniform_prior == 'yes'
    
    if codebook_size != 2:
        raise NotImplementedError("Codebook size other than 2 is not supported")
        
    pl.seed_everything(seed, workers=True)
    
    # get data path
    if env_name == 'toy':
        file_name = f"ppo_all_env_experience.npz"
        # file_name = f"ppo_const_env_experience.npz"
    elif env_name == 'pong':
        file_name = f"pong_data.hdf5"
    else:
        raise NotImplementedError(f"Unknown env_name: {env_name}")
    data_path = '../../data/'
    data_path = os.path.join(data_path, file_name)
    
    # init dataset and dataloader
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
        get_player_pos=get_player_pos,
    )

    if env_name == 'toy':
        train_data, val_data = construct_toy_train_val_data(data_path, **data_kwargs)
    elif env_name == 'pong':
        train_data, val_data = construct_pong_train_val_data(data_path, **data_kwargs)
            
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    # emission settings    
    emission_kwargs = dict(
        num_views=num_views,
        codebook_size=codebook_size,
        embedding_dim=embedding_dim,
        num_variables=num_variables,
        sparse=sparsemax,
        img_shape=train_data.img_shape,
        scale=obs_scale,
        kernel_size=kernel_size,
        depth=depth,
    )
    
    num_values = 1
    vp_kwargs = dict(
        output_dim=num_values,
        mlp_hidden_dims=vp_layer_dims,
        vp_batchnorm=vp_batchnorm,
    )
    
    num_actions = {'pong': 6, 'toy': 4}[env_name]
    
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
        view_masks=train_data.mvwrapper.view_masks,
        disable_recon_loss=disable_recon_loss,
        sparsemax=sparsemax,
        sparsemax_k=sparsemax_k,
        disable_vp=disable_vp,
        action_layer_dims=action_layer_dims,
        kl_scaling=kl_scaling,
        force_uniform_prior=force_uniform_prior,
        prior_noise_scale=prior_noise_scale,
        nonzero_thresh=nonzero_thresh,
    )
    
    # set up wandb
    wandb_config = dict(
        seed=seed,
        codebook_size=codebook_size,
        num_variables=num_variables,
        embedding_dim=embedding_dim,
        kl_balancing_coeff=kl_balancing_coeff,
        l_unroll=l_unroll,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        gradient_clip_val=gradient_clip_val,
        num_views=num_views,
        percentage=percentage,
        dropout=dropout,
        test_only_dropout=test_only_dropout,
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
        kernel_size=kernel_size,
        depth=depth,
        nonzero_thresh=nonzero_thresh,
    )
    
    wandb_proj = "ToyTask" if env_name == 'toy' else "Pong"
    
    logger = WandbLogger(project=f"MT-{wandb_proj}-Ours", config=wandb_config, group=wandb_group, id=wandb_id)
    
    # callbacks
    callbacks = []
    callbacks.append(pl.callbacks.ModelCheckpoint(save_top_k=1, verbose=True))
    callbacks.append(pl.callbacks.TQDMProgressBar(refresh_rate=1))
    callbacks.append(ExtrapolateCallback(dataset=val_data, every_n_batches=100))
    callbacks.append(ReconstructionCallback(dataset=train_data, every_n_batches=100))
    
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
        deterministic=True
    )
    # start training
    trainer.fit(model, train_loader, val_loader)
    
    # finish logging
    logger.experiment.finish()

if __name__ == '__main__':
    parser = create_train_parser()
    args = parser.parse_args()

    main(**vars(args))
    
