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

RewardSupport = namedtuple("RewardSupport", ["min", "max", "size"])

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
        obs, actions, vp, terms, dropped = dataset.get_no_drop(0)
        self.obs = torch.from_numpy(obs)
        self.actions = torch.from_numpy(actions[None])
        self.dropped = torch.from_numpy(dropped[None])
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
    
    # def on_epoch_end(self, trainer, pl_module):
    #     """
    #     This function is called after every epoch.
    #     Call the save_and_sample function every N epochs.
    #     """
    #     self.extrapolate(trainer, pl_module, pl_module.global_step+1)

    @torch.no_grad()
    def extrapolate(self, trainer, pl_module, epoch):
        if self.obs.device != pl_module.device:
            self.obs = self.obs.to(pl_module.device)
            self.actions = self.actions.to(pl_module.device)
            self.dropped = self.dropped.to(pl_module.device)
        
        self.prior = pl_module.prior(1) # batch size 1
        
        if pl_module.hparams.sparsemax:
            self.prior, self.state_idcs = sparsemax_k(self.prior[0], pl_module.hparams.sparsemax_k, pl_module.transition.bitconverter) 
            #TODO add batch support
            self.prior = self.prior[None]
            
            # print(f"{state_belief.shape = }")
            # print(f"{state_idcs.shape = }")
        else:
            state_logits = F.log_softmax(self.prior, dim=-1)
            temp = state_logits[:,0]
            for i in range(1, state_logits.shape[1]):
                temp = temp[...,None] + state_logits[:,i,None,:]
                temp = torch.flatten(temp, start_dim=1)
            self.prior = F.softmax(temp, dim=-1)
            self.state_idcs = None
        
        # extrapolations
        ent = -(self.prior * self.prior.log())
        ent[self.prior == 0] = 0
        ent = ent.sum()
        print(f"0: prior entropy: {ent:.4f}")
        
        self.posterior_0, _ = pl_module.network.compute_posterior(self.prior, self.state_idcs, self.obs[0,None], self.dropped[:,0])
        ent = -(self.posterior_0 * self.posterior_0.log())
        ent[self.posterior_0 == 0] = 0
        ent = ent.sum()
        print(f"0: post  entropy: {ent:.4f}")
        
        state_belief_prior_sequence, state_bit_vec_sequence = pl_module.network.k_step_extrapolation(self.posterior_0, self.state_idcs, self.actions, self.actions.shape[1])
        state_belief_prior_sequence = torch.cat([self.posterior_0[:,None], state_belief_prior_sequence], dim=1)
        if state_bit_vec_sequence is not None:
            state_bit_vec_sequence = torch.cat([self.state_idcs[None], state_bit_vec_sequence], dim=0)
        
        for i in range(1,state_belief_prior_sequence.shape[1]):
            ent = -(state_belief_prior_sequence[:,i] * state_belief_prior_sequence[:,i].log())
            ent[state_belief_prior_sequence[:,i] == 0] = 0
            ent = ent.sum()
            print(f"{i}: prior entropy: {ent:.4f}")
        # displ = state_belief_prior_sequence
        # displ[displ < 0.5] = 0
        # displ[displ > 0.5] = 1
        # displ = displ.reshape(displ.shape[0], displ.shape[1], 7,7)
        # print(displ)
        # print(f"{state_belief_prior_sequence = }")
        # print(f"{state_idcs_prior_sequence = }")
        
        
        emission_input = einops.rearrange(state_belief_prior_sequence, 'b t ... -> (b t) ...')
        
        obs_hat = pl_module.emission.decode_only(emission_input, state_bit_vec_sequence).to('cpu').float()
        num_views = self.obs.shape[1]
        images = torch.stack(
            [(self.obs[:,i].to('cpu')*self.sigma)+self.mu for i in range(num_views)]\
            + [(obs_hat[:-1,i].to('cpu')*self.sigma)+self.mu for i in range(num_views)],
            dim = 1
        ).reshape(2*num_views*self.obs.shape[0], 1, self.obs.shape[2], self.obs.shape[3])
        # log images
        pl_module.logger.experiment.log({'Extrapolation': wandb.Image(make_grid(images, nrow=2*num_views))})

        
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
):
    print(sparsemax)
    print(type(sparsemax))
    if sparsemax:
        logging.warning("\n\n\nSparsemax active: Overriding number of variables to 16!!!\n\n\n")
        num_variables = 16
        
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
        train_val_split = 0.9,
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
    )
    
    num_values = 1
    mlp_hidden_dims = [128, 128]
    vp_kwargs = dict(
        num_values=num_values,
        mlp_hidden_dims=mlp_hidden_dims,
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
    )
    wandb_kwargs = dict(project="MT-ToyTask-Ours", config=wandb_config)
    logger = WandbLogger(**wandb_kwargs)
    
    # callbacks
    callbacks = []
    callbacks.append(pl.callbacks.ModelCheckpoint(save_top_k=1, verbose=True))
    callbacks.append(pl.callbacks.TQDMProgressBar(refresh_rate=1))
    callbacks.append(ExtrapolateCallback(dataset=val_data, every_n_batches=50))
    
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
    parser.add_argument('--test_only_dropout', default=False, type=bool)
    parser.add_argument('--max_datapoints', type=int, default=None)
    
    ## model args
    parser.add_argument('--kl_balancing_coeff', type=float, default=0.8)
    parser.add_argument('--kl_scaling', type=float, default=0.1)
    parser.add_argument('--l_unroll', type=int, default=1)
    parser.add_argument('--discount_factor', type=float, default=0.99)
    parser.add_argument('--num_variables', type=int, default=10)
    parser.add_argument('--codebook_size', type=int, default=2)
    parser.add_argument('--embedding_dim', type=int, default=128)
    parser.add_argument('--mlp_repr', default=False, type=bool)
    parser.add_argument('--disable_recon_loss', default=False, type=bool)
    parser.add_argument('--sparsemax', default=False, type=bool)
    parser.add_argument('--sparsemax_k', type=int, default=30)
    parser.add_argument('--disable_vp', default=False, type=bool)
    parser.add_argument('--action_layer_dims', type=int, nargs='*', default=None)
    
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
    parser.add_argument('--max_len', type=int, default=20, help='Max length of an episode. Longer episodes will be discarded')
    
    args = parser.parse_args()
    
    main(**vars(args))
    
