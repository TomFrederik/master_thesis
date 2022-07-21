import logging
from typing import Tuple, Union
import sys

sys.path.append('../../')

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.common import Tensor
from src.ours.emission import EmissionModel
from src.ours.sparsemax_k import BitConverter, sparsemax_k
from src.ours.state_prior import StatePrior
from src.ours.transition_models import FactorizedTransition
from src.ours.utils import discrete_entropy, kl_balancing_loss
from src.ours.value_prefix import MarginalValuePrefixPredictor


class DiscreteNet(nn.Module):
    def __init__(
        self,
        state_prior: StatePrior,
        transition,
        value_prefix_predictor: MarginalValuePrefixPredictor,
        emission,
        kl_balancing_coeff: float,
        l_unroll: int,
        discount_factor: float,
        reward_support,
        view_masks,
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
        self.value_prefix_predictor: Union[MarginalValuePrefixPredictor, MarginalValuePrefixPredictor] = value_prefix_predictor
        self.disable_vp = value_prefix_predictor is None
        self.emission: EmissionModel = emission
        self.kl_balancing_coeff = kl_balancing_coeff
        self.l_unroll = l_unroll
        self.discount_factor = discount_factor
        self.reward_support = reward_support
        self.view_masks = view_masks
        self.disable_recon_loss = disable_recon_loss
        self.sparsemax = sparsemax
        self.sparsemax_k = sparsemax_k
        self.kl_scaling = kl_scaling
        self.force_uniform_prior = force_uniform_prior
        
        self.recon_normalizer = nn.Parameter(data=torch.stack([torch.from_numpy(view_masks[i]) for i in range(len(view_masks))], dim=0).sum(dim=[1,2]), requires_grad=False)

        self.discount_array = self.discount_factor ** torch.arange(self.l_unroll)
        
        if self.sparsemax:
            self.bitconverter = BitConverter(state_prior.num_variables, 'cuda')
        
    # def compute_value_prefixes(self, reward_sequence):
    #     if self.discount_array.device != reward_sequence.device:
    #         self.discount_array.to(reward_sequence.device)
    #     reward_sequence = torch.cat([reward_sequence, torch.zeros((reward_sequence.shape[0], self.l_unroll-1), device=reward_sequence.device)], dim=1)
    #     return F.conv1d(reward_sequence[:,None], self.discount_array[None,None,:])[:,0]

    def compute_posterior(self, prior: Tensor, state_bit_vecs: Tensor, obs_frame: Tensor, dropped: Tensor) -> Tuple[Tensor, Tensor]:
        obs_logits = self.emission(obs_frame, state_bit_vecs, self.view_masks)
        obs_probs = F.softmax(obs_logits, dim=1) # (batch, num_states, num_views)
        posterior = prior
        for view in range(obs_probs.shape[-1]):
            update = obs_probs[...,view] * (1 - dropped)[:, None, view]
            # print(f"update {view}: {update = }")
            nonzeros = torch.sum(update > 1e-6, dim=1)
            # this is a bit hacky but the idea is that the condition evaluates to true if the view is dropped and the update is 0
            # this means that if the view is not dropped (but the update is 0) the posterior is still 0 for that state
            posterior = torch.where((update + (1-dropped)[:, None, view]) == 0, posterior, update * posterior)
        
        posterior = posterior / posterior.sum(dim=-1, keepdim=True)
        return posterior, obs_logits

    def forward(self, obs_sequence, action_sequence, value_prefix_sequence, nonterms, dropped, player_pos):
        outputs = dict()
        batch_size, seq_len, channels, h, w = obs_sequence.shape
        dimension = h * w 
        if not self.disable_vp:
            target_value_prefixes = value_prefix_sequence
        else:
            target_value_prefixes = None
            
        # prior
        state_belief = self.state_prior(batch_size)
        if self.sparsemax:
            state_belief, state_bit_vecs = sparsemax_k(state_belief, self.sparsemax_k) 
        else:
            state_logits = F.log_softmax(state_belief, dim=-1)
            print(f"{state_logits.shape = }")
            raise ValueError
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

        # dynamics
        posterior_belief_sequence, posterior_bit_vec_sequence, prior_belief_sequence, prior_entropies, posterior_entropies, obs_logits_sequence, value_prefix_pred, dyn_loss = self.process_sequence(
            action_sequence, 
            dropped, 
            posterior_0, 
            state_bit_vecs, 
            obs_sequence,
            nonterms,
            player_pos,
        )
        prior_belief_sequence = torch.cat([state_belief[:,None], prior_belief_sequence], dim=1)
        
        obs_logits_sequence = torch.cat([obs_logits_0[:,None], obs_logits_sequence], dim=1)

        # compute losses
        recon_loss = self.compute_recon_loss(posterior_belief_sequence, obs_logits_sequence, dropped, nonterms)
        
        if not self.disable_vp:
            value_prefix_loss = self.compute_vp_loss(value_prefix_pred, target_value_prefixes) # TODO if we use longer rollout need to adapt this
        else:
            value_prefix_loss = 0
        
        outputs['prior_loss'] = self.kl_scaling * prior_loss * int(not self.force_uniform_prior)
        outputs['value_prefix_loss'] = value_prefix_loss
        outputs['recon_loss'] = recon_loss
        outputs["dyn_loss"] = self.kl_scaling * dyn_loss / self.l_unroll
        outputs['prior_entropy'] = (prior_entropy + sum(prior_entropies))/(len(prior_entropies) + 1)
        outputs['posterior_entropy'] = (posterior_entropy + sum(posterior_entropies))/(len(posterior_entropies) + 1)
        outputs['posterior_belief_sequence'] = posterior_belief_sequence
        outputs['posterior_bit_vec_sequence'] = posterior_bit_vec_sequence
        
        return outputs

    def prepare_vp_input(self, belief, bit_vecs):
        if self.sparsemax:
            # return (belief[...,None] * bit_vecs[None]).flatten(start_dim=1)
            return (belief, bit_vecs)
        else:
            return (belief, )

    def process_sequence(
        self, 
        action_sequence, 
        dropped, 
        state_belief, 
        state_bit_vecs, 
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
        prior_belief_sequence = []
        
        if self.l_unroll > 1:
            raise ValueError("l_unroll > 1 not implemented -> posterior update will not work as intended, also value prefix is not gonna work")

        for t in range(1, action_sequence.shape[1]):
            # predict value prefixes
            if not self.disable_vp:
                vp_input = self.prepare_vp_input(posterior_belief_sequence[-1], posterior_bit_vecs_sequence[-1])
                if self.sparsemax:
                    # value_prefix_pred.append(torch.einsum('ij,ij->i', posterior_belief_sequence[-1], self.value_prefix_predictor(*vp_input)))
                    value_prefix_pred.append(self.value_prefix_predictor(*vp_input))
                else:
                    value_prefix_pred.append(self.value_prefix_predictor(*vp_input))
                    
            
            # get the priors for the next state
            prior, state_bit_vecs = self.transition(posterior_belief_sequence[-1], action_sequence[:,t-1], posterior_bit_vecs_sequence[-1])

            # numerical stability
            # NOTE: do I actually still need this?
            prior = prior + 1e-8
            prior = prior / torch.sum(prior, dim=-1, keepdim=True)

            # get the posterior for the next state
            state_belief_posterior, obs_logits = self.compute_posterior(prior, state_bit_vecs, obs_sequence[:,t], dropped[:,t])
            
            # compute the dynamics loss
            dyn_loss = dyn_loss + kl_balancing_loss(self.kl_balancing_coeff, prior, state_belief_posterior, nonterms[:,t])

            # log            
            posterior_belief_sequence.append(state_belief_posterior)
            posterior_bit_vecs_sequence.append(state_bit_vecs)
            obs_logits_sequence.append(obs_logits)
            prior_belief_sequence.append(prior)
            if nonterms[:,t].sum() > 0:
                prior_entropies.append((discrete_entropy(prior) * nonterms[:,t]).sum() / nonterms[:,t].sum())
                posterior_entropies.append((discrete_entropy(state_belief_posterior) * nonterms[:,t]).sum() / nonterms[:,t].sum())
            else:
                prior_entropies.append(torch.zeros_like(prior_entropies[-1]))
                posterior_entropies.append(torch.zeros_like(posterior_entropies[-1]))
        
        # predict value prefixes
        vp_input = self.prepare_vp_input(posterior_belief_sequence[-1], posterior_bit_vecs_sequence[-1])
        if self.sparsemax:
            # value_prefix_pred.append(torch.einsum('ij,ij->i', posterior_belief_sequence[-1], self.value_prefix_predictor(*vp_input)))
            value_prefix_pred.append(self.value_prefix_predictor(*vp_input))
        else:
            value_prefix_pred.append(self.value_prefix_predictor(*vp_input))
        
        # stack along time dimension
        obs_logits_sequence = torch.stack(obs_logits_sequence, dim=1)
        posterior_belief_sequence = torch.stack(posterior_belief_sequence, dim=1)
        prior_belief_sequence = torch.stack(prior_belief_sequence, dim=1)
        if posterior_bit_vecs_sequence[-1] is not None:
            posterior_bit_vecs_sequence = torch.stack(posterior_bit_vecs_sequence, dim=1)
        else:
            posterior_bit_vecs_sequence = None
        if not self.disable_vp:
            value_prefix_pred = torch.stack(value_prefix_pred, dim=1)
        else:
            value_prefix_pred = None
        
        return posterior_belief_sequence, posterior_bit_vecs_sequence, prior_belief_sequence, prior_entropies, posterior_entropies, obs_logits_sequence, value_prefix_pred, dyn_loss

    def compute_recon_loss(self, posterior_belief_sequence, obs_logits_sequence, dropped, nonterms):
        # posterior_belief_sequence has shape (batch, seq_len, num_states)
        # obs_logits_sequence has shape (batch, seq_len, num_active_state, num_views)
        # dropped has shape (batch, seq_len, num_views)
        recon_loss = 0
        if not self.disable_recon_loss:
            recon_loss = (-(1-dropped) * (posterior_belief_sequence[...,None] * obs_logits_sequence).sum(dim=2) * nonterms[...,None])
            recon_loss = recon_loss / self.recon_normalizer[None,None]
            recon_loss = recon_loss.sum(dim=-1)
            num_non_dropped = (1-dropped).sum(dim=-1)
            recon_loss[num_non_dropped > 0] = recon_loss[num_non_dropped > 0] / num_non_dropped[num_non_dropped > 0]
            recon_loss = recon_loss.sum(dim=-1).mean()
        return recon_loss

    def compute_vp_loss(self, value_prefix_pred, target_value_prefixes):
        print(f"{value_prefix_pred = }")
        print(f"{target_value_prefixes = }")
        return F.mse_loss(value_prefix_pred, target_value_prefixes)

    def k_step_extrapolation(self, state_belief, state_bit_vec, action_sequence, k=None):
        if k is None:
            k = self.l_unroll
        state_belief_prior_sequence = []
        state_bit_vec_sequence = []
        for t in range(min(action_sequence.shape[1], k)):
            state_belief, state_bit_vec = self.transition(state_belief, action_sequence[:, t], state_bit_vec)
            state_belief = state_belief / state_belief.sum(dim=-1, keepdim=True)
            state_belief_prior_sequence.append(state_belief)
            state_bit_vec_sequence.append(state_bit_vec)

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
        view_masks,
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
        
        if force_uniform_prior and prior_noise_scale == 0 and sparsemax:
            logging.warning("Using sparsemax + uniform prior + prior_noise_scale 0 is probably not going to work!.")
        #TODO factorize num_variables etc. to be top-level parameters
        self.prior = StatePrior(emission_kwargs['num_variables'], emission_kwargs['codebook_size'], device, force_uniform_prior, prior_noise_scale)
        self.transition = FactorizedTransition(emission_kwargs['num_variables'], emission_kwargs['codebook_size'], num_actions, layer_dims=action_layer_dims, sparse=sparsemax)
        
        if not disable_vp:
            if sparsemax:
                state_size = sparsemax_k * (emission_kwargs['num_variables'])
            else:
                state_size = emission_kwargs['codebook_size'] ** emission_kwargs['num_variables']
            
            self.value_prefix_predictor = MarginalValuePrefixPredictor(emission_kwargs['num_variables'], **vp_kwargs)
        else:
            self.value_prefix_predictor = None
        
        print(self.value_prefix_predictor)
        
        
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
            view_masks,
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
        self.log(f"Training/tuning_loss", total_loss - outputs['dyn_loss'] + 0.01 * unscaled_dyn_loss - outputs['prior_loss'] + 0.01 * unscaled_prior_loss)
        self.log(f"Training/RewPlusUnscDyn", outputs['value_prefix_loss'] + unscaled_dyn_loss)
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

    
