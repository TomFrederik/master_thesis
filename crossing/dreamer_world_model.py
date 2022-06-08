import math

import einops
import pytorch_lightning as pl
import torch
import torch.nn as nn
from dreamerv2.models.dense import DenseModel
from dreamerv2.models.pixel import ObsDecoder, ObsEncoder
from dreamerv2.models.rssm import RSSM


class DreamerWorldModel(pl.LightningModule):
    def __init__(
        self, 
        config,
        device,
    ):
        super().__init__()
        
        self.save_hyperparameters()
        
        self.config = config
        self.action_size = config.action_size
        self.kl_info = config.kl_info
        self.batch_size = config.batch_size
        self.loss_scale = config.loss_scale

        self.RSSM = RSSM(self.action_size, config.rssm_node_size, config.num_views * config.embedding_size, device, config.rssm_type, config.rssm_info)
        
        category_size = config.rssm_info['category_size']
        class_size = config.rssm_info['class_size']
        stoch_size = category_size*class_size
        
        deter_size = config.rssm_info['deter_size']
        modelstate_size = stoch_size + deter_size
        
        self.reward_predictor = DenseModel((1,), modelstate_size, config.reward_config) 
        
        self.ObsEncoders = nn.ModuleList([ObsEncoder(config.obs_shape, config.embedding_size, config.obs_encoder) for _ in range(config.num_views)])
        self.ObsDecoders = nn.ModuleList([ObsDecoder(config.obs_shape, modelstate_size, config.obs_decoder) for _ in range(config.num_views)])

    def compute_train_metrics(self, obs, actions, value_prefixes, terms, dropped):
        train_metrics = dict()

        # convert actions to one-hot vectors
        actions = torch.nn.functional.one_hot(actions, self.action_size)

        nonterms = (1-terms).unsqueeze(-1)

        obs = einops.rearrange(obs, 'b t ... -> t b ...')
        actions = einops.rearrange(actions, 'b t ... -> t b ...')
        nonterms = einops.rearrange(nonterms, 'b t ... -> t b ...')
        value_prefixes = einops.rearrange(value_prefixes, 'b t ... -> t b ...')
        dropped = einops.rearrange(dropped, 'b t ... -> t b ...')

        model_loss, kl_loss, recon_loss, value_prefix_loss, prior_dist, post_dist, posterior = self.representation_loss(obs, actions, value_prefixes, nonterms, dropped)

        with torch.no_grad():
            prior_ent = torch.mean(prior_dist.entropy())
            post_ent = torch.mean(post_dist.entropy())

        train_metrics['model_loss'] = model_loss.mean()
        train_metrics['kl_loss'] = kl_loss.mean()
        train_metrics['recon_loss'] = recon_loss.mean()
        train_metrics['value_prefix_loss'] = value_prefix_loss.mean()
        train_metrics['prior_entropy'] = prior_ent.mean()
        train_metrics['posterior_entropy'] = post_ent.mean()

        return train_metrics

    def forward(self, batch):
        return self.compute_train_metrics(*batch)
    
    def training_step(self, batch, batch_idx):
        train_metrics = self(batch)
        for key, value in train_metrics.items():
            self.log(f"Training/{key}", value)
        return train_metrics['model_loss']

    def validation_step(self, batch, batch_idx):
        train_metrics = self(batch)
        for key, value in train_metrics.items():
            self.log(f"Validation/{key}", value)
        return train_metrics['model_loss']


    def representation_loss(self, obs, actions, value_prefixes, nonterms, dropped):
        embed = torch.cat([enc(obs) for enc in self.ObsEncoders], dim=-1) #t to t+seq_len   
        prev_rssm_state = self.RSSM._init_rssm_state(self.batch_size)   
        prior, posterior = self.RSSM.rollout_observation(len(obs), embed, actions, nonterms, prev_rssm_state)
        post_modelstate = self.RSSM.get_model_state(posterior)               #t to t+seq_len   
        prior_dist, post_dist, div = self._kl_loss(prior, posterior)
        reward_dist = self.reward_predictor(post_modelstate)[...,0]
        reward_loss = self._reward_loss(reward_dist, value_prefixes)
        obs_dist = torch.stack([dec(x) for dec, x in zip(self.ObsDecoders, torch.chunk(post_modelstate, self.config.num_views, -1))], dim=1)
        recon_loss = self._recon_loss(obs_dist, obs, dropped)

        model_loss = self.loss_scale['kl'] * div + recon_loss + reward_loss

        return model_loss, div, recon_loss, reward_loss, prior_dist, post_dist, posterior

    def extrapolate_from_init_obs(self, init_obs, action_sequence):
        embed = embed = self.ObsEncoders[0](init_obs)                                         #t to t+seq_len   
        next_rssm_states = self.RSSM.extrapolate_from_init_obs(embed, action_sequence)
        model_states = self.RSSM.get_model_state(next_rssm_states)
        obs_dist = torch.stack([dec(x) for dec, x in zip(self.ObsDecoders, torch.chunk(model_states, self.config.num_views, -1))], dim=1)
        return obs_dist

    def _reward_loss(self, reward_dist, rewards):
        reward_loss = torch.nn.functional.mse_loss(rewards, reward_dist)
        return reward_loss
    
    def _recon_loss(self, obs_mean, obs, dropped):
        std = 1 # TODO?
        
        obs_mean = einops.rearrange(obs_mean, 't c (h w) -> t 1 c h w ', h=obs.shape[3], w=obs.shape[4])
        exponent = -(obs[dropped == 0] - obs_mean[dropped == 0])**2 / (2*std**2)
        recon_loss = -torch.sum(exponent)
        num_non_dropped = torch.sum(1-dropped)
        if num_non_dropped == 0:
            recon_loss = 0*recon_loss
        else:
            recon_loss = recon_loss / num_non_dropped # average over non missing observations
        return recon_loss
    
    def _kl_loss(self, prior, posterior):
        prior_dist = self.RSSM.get_dist(prior)
        post_dist = self.RSSM.get_dist(posterior)
        if self.kl_info['use_kl_balance']:
            alpha = self.kl_info['kl_balance_scale']
            kl_lhs = torch.mean(torch.distributions.kl.kl_divergence(self.RSSM.get_dist(self.RSSM.rssm_detach(posterior)), prior_dist))
            kl_rhs = torch.mean(torch.distributions.kl.kl_divergence(post_dist, self.RSSM.get_dist(self.RSSM.rssm_detach(prior))))
            if self.kl_info['use_free_nats']:
                free_nats = self.kl_info['free_nats']
                kl_lhs = torch.max(kl_lhs,kl_lhs.new_full(kl_lhs.size(), free_nats))
                kl_rhs = torch.max(kl_rhs,kl_rhs.new_full(kl_rhs.size(), free_nats))
            kl_loss = alpha*kl_lhs + (1-alpha)*kl_rhs

        else: 
            kl_loss = torch.mean(torch.distributions.kl.kl_divergence(post_dist, prior_dist))
            if self.kl_info['use_free_nats']:
                free_nats = self.kl_info['free_nats']
                kl_loss = torch.max(kl_loss, kl_loss.new_full(kl_loss.size(), free_nats))
        return prior_dist, post_dist, kl_loss
    
    def configure_optimizers(self):
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.config.lr, weight_decay=self.config.weight_decay)
        return self.optimizer
