import itertools
import logging
from time import time
from typing import List, Optional, Tuple, Union, TypeVar

import einops
import einops.layers.torch as layers
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F

from commons import Tensor
from sparsemax_k import sparsemax_k, BitConverter
from state_prior import StatePrior
from transition_models import FactorizedTransition
from utils import discrete_entropy, generate_all_combinations, kl_balancing_loss
from value_prefix import DenseValuePrefixPredictor, SparseValuePrefixPredictor
from torch.distributions import OneHotCategorical, Categorical

def sum_factored_logits(logits):
    """
    Sum logits that are factored into shape (*, num_variables, codebook_size).
    """
    num_vars = logits.shape[-2]
    codebook_size = logits.shape[-1]
    out = torch.empty((*logits.shape[:-2], codebook_size ** num_vars), device=logits.device)
    for i, idcs in enumerate(itertools.product(range(codebook_size), repeat=num_vars)):
        out[..., i] = logits[..., torch.arange(len(idcs)), idcs].sum(dim=-1)
    return out


# predicts the value prefix
# takes in sequence (s_t, hat_s_t+1, ..., hat_s_t+k-1)
# and predicts the k step value prefix for each


# combine everythin
class DiscreteNet(nn.Module):
    def __init__(
        self,
        state_prior: StatePrior,
        transition,
        value_prefix_predictor: Union[DenseValuePrefixPredictor, SparseValuePrefixPredictor],
        emission,
        kl_balancing_coeff: float,
        l_unroll: int,
        discount_factor: float,
        reward_support,
        disable_recon_loss: bool = True,
        sparsemax: bool = False,
        sparsemax_k: int = 30,
        kl_scaling: float = 1.0,
        force_uniform_prior: bool = False,
    ):
        if l_unroll < 1:
            raise ValueError('l_unroll must be at least 1')
        
        super().__init__()
        self.state_prior: StatePrior = state_prior
        self.transition: FactorizedTransition = transition
        self.value_prefix_predictor: Union[DenseValuePrefixPredictor, SparseValuePrefixPredictor] = value_prefix_predictor
        self.disable_vp = value_prefix_predictor is None
        self.emission: EmissionModel = emission
        self.kl_balancing_coeff = kl_balancing_coeff
        self.l_unroll = l_unroll
        self.discount_factor = discount_factor
        self.reward_support = reward_support
        self.disable_recon_loss = disable_recon_loss
        self.sparsemax = sparsemax
        self.sparsemax_k = sparsemax_k
        self.kl_scaling = kl_scaling
        self.force_uniform_prior = force_uniform_prior

        self.discount_array = self.discount_factor ** torch.arange(self.l_unroll)
        
        if self.sparsemax:
            self.bitconverter = BitConverter(state_prior.num_variables, 'cuda')
        
    # def compute_value_prefixes(self, reward_sequence):
    #     if self.discount_array.device != reward_sequence.device:
    #         self.discount_array.to(reward_sequence.device)
    #     reward_sequence = torch.cat([reward_sequence, torch.zeros((reward_sequence.shape[0], self.l_unroll-1), device=reward_sequence.device)], dim=1)
    #     return F.conv1d(reward_sequence[:,None], self.discount_array[None,None,:])[:,0]

    def compute_posterior(self, prior: Tensor, state_bit_vecs: Tensor, obs_frame: Tensor, dropped: Tensor) -> Tuple[Tensor, Tensor]:
        obs_logits = self.emission(obs_frame, state_bit_vecs)
        obs_probs = F.softmax(obs_logits, dim=1) # (batch, num_states, num_views)
        posterior = prior
        for view in range(obs_probs.shape[-1]):
            posterior = posterior * obs_probs[...,view] * (1- dropped)[:, None, view]
        posterior = posterior / posterior.sum(dim=-1, keepdim=True)
        return posterior, obs_logits

    def forward(self, obs_sequence, action_sequence, value_prefix_sequence, nonterms, dropped, player_pos):
        outputs = dict()

        batch_size, seq_len, channels, h, w = obs_sequence.shape
        dimension = h * w * channels
        if not self.disable_vp:
            # # compute target value prefixes
            # print(f"{reward_sequence.shape = }")
            # target_value_prefixes = self.compute_value_prefixes(reward_sequence)
            # print(f"{target_value_prefixes.shape = }")
            
            # # convert them to classes
            # transformed_target_value_prefixes = self.scalar_transform(value_prefix_sequence)
            # target_value_prefixes = self.reward_phi(transformed_target_value_prefixes)
            target_value_prefixes = value_prefix_sequence
            # emission model to get emission probs for all possible latent states
            # obs sequence has shape (batch, seq, num_views, 7, 7)
            
            # get vp means for each state
            # value_prefix_means = self.value_prefix_predictor()[:,0] # has shape (num_states, )

        # prior
        state_belief = self.state_prior(batch_size)
        if self.sparsemax:
            #TODO add batch support
            state_belief, state_bit_vecs = sparsemax_k(state_belief, self.sparsemax_k) 
        else:
            state_logits = F.log_softmax(state_belief, dim=-1)
            temp = state_logits[:,0]
            for i in range(1, state_logits.shape[1]):
                temp = temp[...,None] + state_logits[:,i,None,:]
                temp = torch.flatten(temp, start_dim=1)
            state_belief = F.softmax(temp, dim=-1)
            
            state_bit_vecs = None

        # unnormalized p(z_t|x_t)
        posterior_0, obs_logits_0 = self.compute_posterior(state_belief, state_bit_vecs, obs_sequence[:,0], dropped[:,0])
        
        
        # compute entropies
        prior_entropy = discrete_entropy(state_belief).mean()
        posterior_entropy = discrete_entropy(posterior_0).mean()    

        # pull posterior and prior closer together
        prior_loss = kl_balancing_loss(self.kl_balancing_coeff, state_belief, posterior_0, nonterms[:,0])
        state_belief = posterior_0

        # dynamics
        posterior_belief_sequence, posterior_bit_vec_sequence, prior_entropies, posterior_entropies, obs_logits_sequence, value_prefix_pred, dyn_loss = self.process_sequence(
            action_sequence, 
            dropped, 
            posterior_0, 
            state_bit_vecs, 
            obs_sequence,
            nonterms,
            player_pos,
        )
        
        obs_logits_sequence = torch.cat([obs_logits_0[:,None], obs_logits_sequence], dim=1)

        # compute losses
        recon_loss = self.compute_recon_loss(posterior_belief_sequence, posterior_bit_vec_sequence, obs_logits_sequence, dropped, nonterms)
        
        # print(f"{recon_loss = }")
        if not self.disable_vp:
            value_prefix_loss = self.compute_vp_loss(value_prefix_pred, target_value_prefixes) # TODO if we use longer rollout need to adapt this
        else:
            value_prefix_loss = 0
        
        print(value_prefix_pred)
        print(target_value_prefixes)

        outputs['prior_loss'] = self.kl_scaling * prior_loss * int(not self.force_uniform_prior)
        outputs['value_prefix_loss'] = value_prefix_loss 
        outputs['recon_loss'] = recon_loss / dimension
        outputs["dyn_loss"] = self.kl_scaling * dyn_loss / self.l_unroll
        outputs['prior_entropy'] = (prior_entropy + sum(prior_entropies))/(len(prior_entropies) + 1)
        outputs['posterior_entropy'] = (posterior_entropy + sum(posterior_entropies))/(len(posterior_entropies) + 1)
        outputs['posterior_belief_sequence'] = posterior_belief_sequence
        outputs['posterior_bit_vec_sequence'] = posterior_bit_vec_sequence
        
        return outputs

    def prepare_vp_input(self, belief, bit_vecs):
        if self.sparsemax:
            # return belief, bit_vecs.flatten(start_dim=0)[None].float()
            return (belief[...,None] * bit_vecs[None]).flatten(start_dim=1)
            # return torch.cat([belief, bit_vecs.flatten()[None]], dim=-1)
        else:
            return belief

    def process_sequence(
        self, 
        action_sequence, 
        dropped, 
        state_belief, 
        state_bit_vecs, 
        # value_prefix_means, 
        obs_sequence,
        nonterms,
        player_pos,
    ):
        dyn_loss = 0
        value_prefix_pred = []
        posterior_belief_sequence = [state_belief]
        posterior_bit_vecs_sequence = [state_bit_vecs]
        obs_logits_sequence = []
        prior_entropies = []
        posterior_entropies = []
        
        if self.l_unroll > 1:
            raise ValueError("l_unroll > 1 not implemented -> posterior update will not work as intended, also value prefix is not gonna work")

        for t in range(1, action_sequence.shape[1]):
            # predict value prefixes
            if not self.disable_vp:
                if self.sparsemax:
                    vp_input = posterior_bit_vecs_sequence[-1].float()
                    value_prefix_pred.append(torch.einsum('ij,ij->i', posterior_belief_sequence[-1], self.value_prefix_predictor(vp_input)))
                else:
                    vp_input = self.prepare_vp_input(posterior_belief_sequence[-1], posterior_bit_vecs_sequence[-1])
                    value_prefix_pred.append(self.value_prefix_predictor(vp_input))
                    
            
            # get the priors for the next state
            prior, state_bit_vecs = self.transition(posterior_belief_sequence[-1], action_sequence[:,t-1], posterior_bit_vecs_sequence[-1])

            # get the posterior for the next state
            state_belief_posterior, obs_logits = self.compute_posterior(prior, state_bit_vecs, obs_sequence[:,t], dropped[:,t])
            # print(torch.sort(state_belief_posterior, dim=-1)[0][:,-10:])
            # print(torch.sort(state_belief_posterior, dim=-1)[1][:,-10:])
            
            # compute the dynamics loss
            dyn_loss = dyn_loss + kl_balancing_loss(self.kl_balancing_coeff, prior, state_belief_posterior, nonterms[:,t])

            # log            
            posterior_belief_sequence.append(state_belief_posterior)
            posterior_bit_vecs_sequence.append(state_bit_vecs)
            obs_logits_sequence.append(obs_logits)
            if nonterms[:,t].sum() > 0:
                prior_entropies.append((discrete_entropy(prior) * nonterms[:,t]).sum() / nonterms[:,t].sum())
                posterior_entropies.append((discrete_entropy(state_belief_posterior) * nonterms[:,t]).sum() / nonterms[:,t].sum())
            else:
                prior_entropies.append(torch.zeros_like(prior_entropies[-1]))
                posterior_entropies.append(torch.zeros_like(posterior_entropies[-1]))
        # predict value prefixes
        if self.sparsemax:
            vp_input = posterior_bit_vecs_sequence[-1].float()
            value_prefix_pred.append(torch.einsum('ij,ij->i', posterior_belief_sequence[-1], self.value_prefix_predictor(vp_input)))
        else:
            vp_input = self.prepare_vp_input(posterior_belief_sequence[-1], posterior_bit_vecs_sequence[-1])
            value_prefix_pred.append(self.value_prefix_predictor(vp_input))
                    
        
        # for i in range(len(posterior_belief_sequence)):
        #     print(f"{i = } : top-5 = {torch.argsort(posterior_belief_sequence[i][0])[-5:].detach().cpu().numpy()}, ent = {discrete_entropy(posterior_belief_sequence[i]):.3f}")
        
        # stack along time dimension
        obs_logits_sequence = torch.stack(obs_logits_sequence, dim=1)
        posterior_belief_sequence = torch.stack(posterior_belief_sequence, dim=1)
        if posterior_bit_vecs_sequence[-1] is not None:
            posterior_bit_vecs_sequence = torch.stack(posterior_bit_vecs_sequence, dim=1)
        else:
            posterior_bit_vecs_sequence = None
        if not self.disable_vp:
            value_prefix_pred = torch.stack(value_prefix_pred, dim=1)
        else:
            value_prefix_pred = None
        
        return posterior_belief_sequence, posterior_bit_vecs_sequence, prior_entropies, posterior_entropies, obs_logits_sequence, value_prefix_pred, dyn_loss

    def compute_recon_loss(self, posterior_belief_sequence, posterior_bit_vec_sequence, obs_logits_sequence, dropped, nonterms):
        # posterior_belief_sequence has shape (batch, seq_len, num_states)
        # obs_logits_sequence has shape (batch, seq_len, num_active_state, num_views)
        # dropped has shape (batch, seq_len, num_views)
        recon_loss = 0
        if not self.disable_recon_loss:
            recon_loss = ((-(1-dropped)[:,:,None,:] * posterior_belief_sequence[...,None] * obs_logits_sequence).sum(dim=[2,3]) * nonterms).sum(dim=-1).mean()
        return recon_loss

    def compute_vp_loss(self, value_prefix_pred, target_value_prefixes):
        return F.mse_loss(value_prefix_pred, target_value_prefixes)

    def k_step_extrapolation(self, state_belief, state_bit_vec, action_sequence, k=None):
        if k is None:
            k = self.l_unroll
        state_belief_prior_sequence = []
        state_bit_vec_sequence = []
        for t in range(min(action_sequence.shape[1], k)):
            state_belief, state_bit_vec = self.transition(state_belief, action_sequence[:, t], state_bit_vec)
            # print('state_belief in k-step extrapolation: ', state_belief)
            state_belief = state_belief / state_belief.sum(dim=-1, keepdim=True)
            state_belief_prior_sequence.append(state_belief)
            state_bit_vec_sequence.append(state_bit_vec)
        # print(state_belief_prior_sequence)
        # print(state_idcs_prior_sequence)
        if state_bit_vec_sequence[-1] is not None:
            state_bit_vec_sequence = torch.stack(state_bit_vec_sequence, dim=1)
        else:
            state_bit_vec_sequence = None
        return torch.stack(state_belief_prior_sequence, dim=1), state_bit_vec_sequence

    ##########################################################
    # The following functions are from the EfficientZero paper
    # https://github.com/YeWR/EfficientZero/blob/a0c094818d750237d5aa14263a65a1e1e4f2bbcb/core/config.py
    ##########################################################
    def scalar_transform(self, x):
        """ Reference from MuZero: Appendix F => Network Architecture
        & Appendix A : Proposition A.2 in https://arxiv.org/pdf/1805.11593.pdf (Page-11)
        """
        epsilon = 0.001
        sign = torch.ones(x.shape).float().to(x.device)
        sign[x < 0] = -1.0
        output = sign * (torch.sqrt(torch.abs(x) + 1) - 1 + epsilon * x)
        return output

    def reward_phi(self, x):
        return self._phi(x, self.reward_support.min, self.reward_support.max, self.reward_support.size)

    def _phi(self, x, min, max, set_size: int):
        x.clamp_(min, max)
        x_low = x.floor()
        x_high = x.ceil()
        p_high = x - x_low
        p_low = 1 - p_high

        target = torch.zeros(x.shape[0], x.shape[1], set_size).to(x.device)
        x_high_idx, x_low_idx = x_high - min, x_low - min
        target.scatter_(2, x_high_idx.long().unsqueeze(-1), p_high.unsqueeze(-1))
        target.scatter_(2, x_low_idx.long().unsqueeze(-1), p_low.unsqueeze(-1))
        return target
    ##########################################################
    ##########################################################

class LightningNet(pl.LightningModule):
    def __init__(
        self,
        state_dim,
        num_actions,
        vp_kwargs,
        emission_kwargs,
        kl_balancing_coeff,
        l_unroll,
        discount_factor,
        learning_rate,
        weight_decay,
        device,
        reward_support,
        disable_recon_loss = False,
        sparsemax = False,
        sparsemax_k = 30,
        disable_vp = False,
        action_layer_dims = None,
        kl_scaling=1.0,
        force_uniform_prior=False,
        prior_noise_scale=0.0,
    ):
        super().__init__()
        self.save_hyperparameters()
        
        #TODO factorize num_variables etc. to be top-level parameters
        if force_uniform_prior and prior_noise_scale == 0 and sparsemax:
            logging.warning("Using sparsemax + uniform prior + prior_noise_scale 0 is probably not going to work!.")
        self.prior = StatePrior(emission_kwargs['num_variables'], emission_kwargs['codebook_size'], device, force_uniform_prior, prior_noise_scale)
        self.transition = FactorizedTransition(emission_kwargs['num_variables'], emission_kwargs['codebook_size'], num_actions, layer_dims=action_layer_dims, sparse=sparsemax)
        
        if not disable_vp:
            if sparsemax:
                state_size = sparsemax_k * (emission_kwargs['num_variables'])
            else:
                state_size = emission_kwargs['codebook_size'] ** emission_kwargs['num_variables']
            if sparsemax:
                self.value_prefix_predictor = SparseValuePrefixPredictor(emission_kwargs['num_variables'], **vp_kwargs)
            else:
                self.value_prefix_predictor = DenseValuePrefixPredictor(state_size, **vp_kwargs)
            # self.value_prefix_predictor = NewValuePrefixPredictor(49, **vp_kwargs)
            # self.value_prefix_predictor = ValuePrefixPredictor(emission_kwargs['num_variables'], emission_kwargs['codebook_size'], emission_kwargs['embedding_dim'], **vp_kwargs)
        else:
            self.value_prefix_predictor = None
        
        # print(self.value_prefix_predictor)
        # raise ValueError
        
        self.emission = EmissionModel(**emission_kwargs) #TODO
        self.network = DiscreteNet(
            self.prior,
            self.transition,
            self.value_prefix_predictor,
            self.emission,
            kl_balancing_coeff,
            l_unroll,
            discount_factor,
            reward_support,
            disable_recon_loss,
            sparsemax,
            sparsemax_k,
            kl_scaling,
            force_uniform_prior,
        )
    
    def forward(self, obs, actions, value_prefixes, nonterms, dropped, player_pos):
        return self.network(obs, actions, value_prefixes, nonterms, dropped, player_pos)
    
    def training_step(self, batch, batch_idx):
        outputs = self(*batch)
        total_loss = sum(list(val for (key, val) in outputs.items() if key.endswith('loss')))
        for key, value in outputs.items():
            if not key.endswith('sequence'):
                self.log(f"Training/{key}", value)
        unscaled_dyn_loss = outputs['dyn_loss'] / self.hparams.kl_scaling
        unscaled_prior_loss = outputs['prior_loss'] / self.hparams.kl_scaling
        self.log(f'Training/unscaled_dyn_loss', unscaled_dyn_loss)
        self.log(f'Training/unscaled_prior_loss', unscaled_prior_loss)
        self.log(f"Training/total_loss", total_loss)
        self.log(f"Training/total_unscaled_loss", total_loss - outputs['dyn_loss'] + unscaled_dyn_loss - outputs['prior_loss'] + unscaled_prior_loss)
        self.log(f"Training/tuning_loss", total_loss - outputs['dyn_loss'] + 0.01 * unscaled_dyn_loss - outputs['prior_loss'] + unscaled_prior_loss)
        self.log(f"Training/RewPlusUnscDyn", outputs['value_prefix_loss'] + unscaled_dyn_loss)
        if torch.isnan(total_loss).any():
            raise ValueError("Total loss is NaN!")
        return total_loss
    
    def validation_step(self, batch, batch_idx):
        outputs = self(*batch)
        total_loss = sum(list(val for (key, val) in outputs.items() if key.endswith('loss')))
        for key, value in outputs.items():
            if not key.endswith('sequence'):
                self.log(f"Validation/{key}", value)
        unscaled_dyn_loss = outputs['dyn_loss'] / self.hparams.kl_scaling
        unscaled_prior_loss = outputs['prior_loss'] / self.hparams.kl_scaling
        self.log(f'Validation/unscaled_dyn_loss', unscaled_dyn_loss)
        self.log(f'Validation/unscaled_prior_loss', unscaled_prior_loss)
        self.log(f"Validation/total_loss", total_loss)
        self.log(f"Validation/total_unscaled_loss", total_loss - outputs['dyn_loss'] + unscaled_dyn_loss - outputs['prior_loss'] + unscaled_prior_loss)
        self.log(f"Validation/tuning_loss", total_loss - outputs['dyn_loss'] + 0.01 * unscaled_dyn_loss - outputs['prior_loss'] + unscaled_prior_loss)
        self.log(f"Validation/RewPlusUnscDyn", outputs['value_prefix_loss'] + unscaled_dyn_loss)
        
        return total_loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.network.parameters(), lr=self.hparams.learning_rate, weight_decay=self.hparams.weight_decay)
        return optimizer
    

class MLPDecoder(nn.Module):
    def __init__(self, latent_dim, num_vars, output_channels, width=7, scale=1) -> None:
        super().__init__()
        
        self.net = nn.Sequential(
            layers.Rearrange('b n d -> b (n d)'),
            nn.Linear(latent_dim*num_vars, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, output_channels * (width ** 2)),
            layers.Rearrange('b (c h w) -> b c h w', c=output_channels, h=width, w=width),
        )
    
    def forward(self, x):
        return self.net(x)

    def set_bn_eval(self, m):
        if isinstance(m, nn.modules.batchnorm._BatchNorm):
            m.eval()

    def set_bn_train(self, m):
        if isinstance(m, nn.modules.batchnorm._BatchNorm):
            m.train()
            
            
class ResNetBlock(nn.Module):

    def __init__(self, num_in_channels, num_hidden_channels, shape, stride=1):
        super().__init__()
        self.ln1 = nn.LayerNorm([num_in_channels, *shape])
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(num_in_channels, num_hidden_channels, 3, stride, padding=1)
        
        self.ln2 = nn.LayerNorm([num_hidden_channels, *shape])
        self.conv2 = nn.Conv2d(num_hidden_channels, num_in_channels, 3, stride, padding=1)
        
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.ln1(x)
        out = self.relu(out)
        out = self.conv1(out)

        out = self.ln2(out)
        out = self.relu(out)
        out = self.conv2(out)

        out += residual

        return out


### stolen from dreamer
def conv_out(h_in, padding, kernel_size, stride):
    return int((h_in + 2. * padding - (kernel_size - 1.) - 1.) / stride + 1.)

def output_padding(h_in, conv_out, padding, kernel_size, stride):
    return h_in - (conv_out - 1) * stride + 2 * padding - (kernel_size - 1) - 1

def conv_out_shape(h_in, padding, kernel_size, stride):
    return tuple(conv_out(x, padding, kernel_size, stride) for x in h_in)

class Decoder(nn.Module):

    def __init__(
        self, 
        latent_dim: int = 32, 
        num_vars: int = 32, 
        output_channels: int = 3, 
        depth: int = 16, 
        scale: int = 1, 
        kernel_size: int = 3
    ) -> None:
        super().__init__()
        # dreamer
        output_shape = (1, 7*scale, 7*scale)
        c, h, w = output_shape

        d = depth
        k  = kernel_size
        conv1_shape = conv_out_shape(output_shape[1:], 0, k, 1)
        conv2_shape = conv_out_shape(conv1_shape, 0, k, 1)
        conv3_shape = conv_out_shape(conv2_shape, 0, k, 1)
        self.conv_shape = (4*d, *conv3_shape)
        self.output_shape = output_shape
        
        self.linear = nn.Linear(latent_dim*num_vars, np.prod(self.conv_shape).item())
        
        self.net = nn.Sequential(
            layers.Rearrange('b n d -> b (n d)'),
            self.linear,
            layers.Rearrange('b (d h w) -> b d h w', d=self.conv_shape[0], h=self.conv_shape[1], w=self.conv_shape[2]),
            nn.ConvTranspose2d(4*d, 2*d, k, 1),
            nn.ELU(alpha=1.0),
            nn.ConvTranspose2d(2*d, d, k, 1),
            nn.ELU(alpha=1.0),
            nn.ConvTranspose2d(d, output_channels, k, 1),
            # layers.Rearrange('b d h w -> b (d h w)'),
            # nn.Linear(16*49, output_channels*49),
            # layers.Rearrange('b (d h w) -> b d h w', h=7, w=7),
        )
        
        # test
        # self.net = nn.Sequential(
        #     layers.Rearrange('b n d -> b (n d)'),
        #     nn.Linear(latent_dim*num_vars, 3*3*128),
        #     layers.Rearrange('b (d h w) -> b d h w', h=3, w=3),
        #     nn.ConvTranspose2d(128, 64, (3,3), (1,1)),
        #     nn.ReLU(),
        #     nn.ConvTranspose2d(64, 32, (3,3), (1,1)),
        #     nn.ReLU(),
        #     nn.ConvTranspose2d(32, 1, (3,3), (1,1)),
        #     # layers.Rearrange('b d h w -> b (d h w)'),
        #     # nn.Linear(16*49, output_channels*49),
        #     # layers.Rearrange('b (d h w) -> b d h w', h=7, w=7),
        # )
        
        
        # # ResNet
        # self.net = nn.Sequential(
        #     layers.Rearrange('b n d -> b (n d)'),
        #     nn.Linear(latent_dim*num_vars, 4*4*16),
        #     layers.Rearrange('b (c h w) -> b c h w', c=16, h=4, w=4),
        #     ResNetBlock(16, 64, shape=(4,4)),
        #     nn.ConvTranspose2d(16, 64, (2,2), (1,1)),
        #     nn.ReLU(),
        #     ResNetBlock(64, 64, shape=(5,5)),
        #     nn.ConvTranspose2d(64, 16, (2,2), (1,1)),
        #     nn.ReLU(),
        #     ResNetBlock(16, 16, shape=(6,6)),
        #     nn.ConvTranspose2d(16, 16, (2,2), (1,1)),
        #     nn.ReLU(),
        #     layers.Rearrange('b d h w -> b (d h w)'),
        #     nn.Linear(16*49, output_channels*49),
        #     layers.Rearrange('b (d h w) -> b d h w', h=7, w=7),
        # )

        # ours
        # self.net = nn.Sequential(
        #     layers.Rearrange('b n d -> b (n d)'),
        #     nn.Linear(latent_dim*num_vars, 2048),
        #     layers.Rearrange('b (d h w) -> b d h w', h=4, w=4),
        #     nn.ConvTranspose2d(128, 64, (2,2), (1,1)),
        #     nn.ReLU(),
        #     nn.ConvTranspose2d(64, 32, (2,2), (1,1)),
        #     nn.ReLU(),
        #     nn.ConvTranspose2d(32, output_channels, (2,2), (1,1)),
        # )
        
        # init
        # torch.nn.init.kaiming_uniform_(self.net[3].weight, mode='fan_out')
        # torch.nn.init.kaiming_uniform_(self.net[5].weight, mode='fan_out')
        
        
    def forward(self, x):        
        out = self.net(x)
        return out
        

    def set_bn_eval(self, m):
        if isinstance(m, nn.modules.batchnorm._BatchNorm):
            m.eval()


    def set_bn_train(self, m):
        if isinstance(m, nn.modules.batchnorm._BatchNorm):
            m.train()

class EmissionModel(nn.Module):

    def __init__(
        self, 
        num_input_channels: int,
        embedding_dim: int,
        codebook_size: int,
        num_variables: int,
        mlp: Optional[bool] = False,
        sparse: Optional[bool] = False,
        scale: int = 1,
        kernel_size: int = 3,
        depth: int = 16,
    ):
        super().__init__()
        self.num_input_channels = num_input_channels
        self.embedding_dim = embedding_dim
        self.codebook_size = codebook_size
        self.num_variables = num_variables
        self.sparse = sparse
        self.kernel_size = kernel_size
        
        if mlp:
            self.decoders = nn.ModuleList([MLPDecoder(embedding_dim, num_variables, output_channels=1, width=7, scale=scale) for _ in range(num_input_channels)])
        else:
            self.decoders = nn.ModuleList([Decoder(embedding_dim, num_variables, output_channels=1, scale=scale, kernel_size=kernel_size, depth=depth) for _ in range(num_input_channels)])
        self.latent_embedding = nn.Parameter(torch.zeros(num_variables, codebook_size, embedding_dim))
        
        nn.init.normal_(self.latent_embedding)
        
        print(self)
        if not sparse:
            self.all_idcs = generate_all_combinations(codebook_size, num_variables)

    def forward(
        self, 
        x, 
        state_bit_vecs: Optional[Tensor] = None
    ) -> Tensor:
        if self.sparse:
            if state_bit_vecs is None:
                raise ValueError('state_bit_vecs must be provided for sparse model')
            emission_means = self.get_emission_means_sparse(state_bit_vecs[:,None])
            obs_logits = self.compute_obs_logits_sparse(x, emission_means[:,0])
        else:
            # assuming a diagonal gaussian with unit variance
            emission_means = self.get_emission_means()
            obs_logits = self.compute_obs_logits(x, emission_means)
        return obs_logits

    def get_emission_means_sparse(
        self, 
        state_bit_vecs: Tensor, 
    ) -> Tensor:
        states = F.one_hot(state_bit_vecs, num_classes=2).float()
        embeds = torch.einsum("btkdc,dce->btkde", states, self.latent_embedding)
        batch, time, k, *_ = embeds.shape
        embeds = einops.rearrange(embeds, 'batch time k vars dim -> (batch time k) vars dim')
        emission_probs = torch.stack([decoder(embeds)[:,0] for decoder in self.decoders], dim=1)
        emission_probs = einops.rearrange(emission_probs, '(batch time k) views h w -> batch time k views h w', batch=batch, k=k, time=time)
        return emission_probs

    def get_emission_means(self):
        z = self.latent_embedding[torch.arange(self.num_variables),self.all_idcs,:]
        emission_probs = torch.stack([decoder(z)[:,0] for decoder in self.decoders], dim=1)
        return emission_probs
        
    def compute_obs_logits_sparse(self, x, emission_means):
        #TODO separate channels and views rather than treating them interchangably?
        output = - ((emission_means - x[:,None]) ** 2).sum(dim=[-2,-1]) / 2
        return output
    
    def compute_obs_logits(self, x, emission_means):
        #TODO separate channels and views rather than treating them interchangably?
        output = - ((emission_means[None] - x[:,None]) ** 2).sum(dim=[-2,-1]) / 2
        return output

    
    ## hardcoded for numvars = 1, codebooksize 49
    # def compute_obs_logits(self, x, emission_means):
    #     #TODO separate channels and views rather than treating them interchangably?
    #     output = einops.rearrange(torch.zeros_like(x), '... views h w -> ... views (h w)') - 10 ** 10
    #     output[torch.arange(len(x)),...,torch.argmin(einops.rearrange(x, '... views h w -> ... (views h w)'), dim=-1)] = 0
    #     return output
    
    
    @torch.no_grad()
    def decode_only(self, latent_dist, bit_vecs=None):
        """

        :param latent_dist: shape (batch_size, codebook_size ** num_vars)
        :type latent_dist: torch.Tensor
        :return: x_hat: decoded z
        :rtype: torch.Tensor
        """
        mean_prediction = None

        # TODO make this more efficient using batching
        # compute expected value
        if self.sparse:
            emission_means = self.get_emission_means_sparse(bit_vecs)
            mean_prediction = (latent_dist[...,None,None,None] * emission_means).sum(dim=2)
        else:
            emission_means = self.get_emission_means()
            mean_prediction = (latent_dist[...,None,None,None] * emission_means[None]).sum(dim=2)
                

        return mean_prediction

    @property
    def device(self):
        return next(self.parameters()).device
    


    
