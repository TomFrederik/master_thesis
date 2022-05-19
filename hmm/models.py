from collections import deque
import itertools
import logging
import math
from typing import Optional

import einops
import einops.layers.torch as layers
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
import numpy as np
import matplotlib.pyplot as plt
import typing

from utils import discrete_kl, batched_query_attention, mem_efficient_attention, discrete_entropy
from sparsemax_k import sparsemax_k, sparse_transition, convert_state_idx_to_bit_vector

Tensor = typing.TypeVar('Tensor', bound=torch.Tensor)

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


# models the prior distribution of the latent space
class StatePrior(nn.Module):
    def __init__(self, num_variables, codebook_size, device):
        super().__init__()
        self.state_dim = codebook_size ** codebook_size

        # init prior to uniform distribution
        self.prior = nn.Parameter(torch.zeros((num_variables, codebook_size), device=device))
        
    def forward(self, batch_size):
        return einops.repeat(self.prior, 'num_vars codebook -> batch_size num_vars codebook', batch_size=batch_size)

    def to(self, device):
        self.prior = self.prior.to(device)
    
    @property
    def device(self):
        return self.prior.device

# models the transition distribution of the latent space conditioned on the action
# instantiates |A| transition matrices
class ActionConditionedTransition(nn.Module):
    def __init__(self, num_states, num_actions, device, factorized=False):
        super().__init__()
        self.num_states = num_states
        self.num_actions = num_actions
        self.factorized = factorized
        
        if factorized:
            raise NotImplementedError
        else:
            self.matrices = nn.Parameter(torch.zeros((self.num_actions, self.num_states, self.num_states), device=device))
    
    def forward(self, state, action):
        if self.factorized:
            raise NotImplementedError
        else:
            matrix = self.matrices[action]
            return torch.einsum('bi, bij -> bj', state, torch.softmax(matrix, dim=-1))
    
    def to(self, device):
        self.matrices = self.matrices.to(device)

class ActionConditionedMLPTransition(nn.Module):
    def __init__(self, num_states, num_actions, hidden_dim=128):
        super().__init__()
        self.num_states = num_states
        self.num_actions = num_actions
        
        self.action_emb = nn.Embedding(num_actions, hidden_dim)
        for m in self.action_emb:
            nn.init.xavier_uniform_(m.weight)
        
        # self.state_emb = nn.Linear(num_states, hidden_dim)
        
        self.mlp = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, num_states**2))
        
        # self.mlp = nn.Sequential(
        #     nn.Linear(hidden_dim*2, hidden_dim*2),
        #     nn.ReLU(),
        #     nn.Linear(hidden_dim*2, num_states)
        # )

    def forward(self, state, action):
        action = self.action_emb(action)
        # state = self.state_emb(state)
        # return torch.softmax(self.mlp(torch.cat([state, action], dim=-1)), dim=-1)
        transition_matrix = einops.rearrange(self.mlp(action), 'batch (inp out) -> batch inp out', inp=state.shape[-1], out=state.shape[-1])
        return torch.einsum('bi, bij -> bj', state, torch.softmax(transition_matrix, dim=-1))

class ActionConditionedFactorizedTransition(nn.Module):
    def __init__(self, num_states, num_actions, embedding_dim=128, hidden_dim=128, layer_dims=None, batch_size=-1):
        super().__init__()
        self.num_states = num_states
        self.num_actions = num_actions
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.layer_dims = None
        
        self.state_emb = nn.Embedding(num_states, embedding_dim)
        nn.init.xavier_uniform_(self.state_emb.weight) # necessary to get uniform distribution at init
        # self.action_emb = nn.Embedding(num_actions, embedding_dim)
        
        self.keys = nn.ModuleList([StateFeatures(embedding_dim, hidden_dim, layer_dims) for _ in range(num_actions)])
        self.queries = nn.ModuleList([StateFeatures(embedding_dim, hidden_dim, layer_dims) for _ in range(num_actions)])
        
        # self.keys = nn.ModuleList([nn.Linear(embedding_dim, hidden_dim, bias=False) for _ in range(self.num_actions)])
        # self.queries = nn.ModuleList([nn.Linear(embedding_dim, hidden_dim, bias=False) for _ in range(self.num_actions)])
    
    
    def _get_transition_matrix(self, action):
        keys = self.keys[action](self.state_emb.weight)
        queries = self.queries[action](self.state_emb.weight)
        return self.compute_attention(keys, queries)
    
    def forward(self, state_belief, state_idcs, action):
        # action = self.action_emb(action)
        # transition_matrix = self._get_transition_matrix(action)
        return sparse_transition(self.state_emb.weight, self.state_emb.weight, self.keys[action], self.queries[action], state_belief, state_idcs)

    def compute_attention(self, keys, queries):
        # out = mem_efficient_attention(keys, queries, self.query_chunk_size, self.key_chunk_size)
        out = mem_efficient_attention(keys, queries, 1024, 4096)
        print(f'{keys.shape = }')
        print(f'{queries.shape = }')
        print(f"{out.shape = }")
        raise ValueError
        return out
        # return batched_query_attention(keys, queries, self.batch_size)

class StateFeatures(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, layer_dims=None):
        super().__init__()
        self.net = nn.Sequential()
        if layer_dims is None:
            layer_dims = [embedding_dim, hidden_dim]
        else:
            layer_dims = [embedding_dim] + layer_dims + [hidden_dim]
        
        for i in range(len(layer_dims)-1):
            self.net.add_module(f"{i}", nn.Linear(layer_dims[i], layer_dims[i+1]))
            if i != len(layer_dims)-2:
                self.net.add_module(f"{i}_relu", nn.ReLU())
        
    def forward(self, state):
        return self.net(state)

# class ActionConditionedMLPTransition(nn.Module):
#     def __init__(self, state_dim, num_actions, hidden_dim=128):
#         super().__init__()
#         self.state_dim = state_dim
#         self.num_actions = num_actions

#         self.wall_idcs = [(0,3),(1,3),(2,3),(3,3),(4,3),(6,3)]
        
#         # if action == 0: # down
#         #     next_cell[0] += 1
#         # elif action == 1: # up
#         #     next_cell[0] -= 1
#         # elif action == 2: # right
#         #     next_cell[1] += 1
#         # elif action == 3: # left
#         #     next_cell[1] -= 1
#         test = torch.zeros(49).to('cuda')
#         test[43] = 1
#         print(test.reshape((7,7)))

#         down = torch.diag_embed(torch.ones(self.state_dim), offset=-7).to('cuda')
#         down = down[:-7, :-7]
#         down[-7:,-7:] = torch.diag_embed(torch.ones(7)).to('cuda')
#         print((down @ test).reshape((7,7)))

#         right = torch.diag_embed(torch.ones(self.state_dim), offset=-1).to('cuda')
#         right = right[:-1, :-1]
#         right[[6,13,20,27,34,41,48], [6,13,20,27,34,41,48]] = 1
        
#         for (row, col) in self.wall_idcs:
#             idx = col + row * 7
#             right[:,idx-1] = 0
#             right[idx-1,idx-1] = 1
#         print((right @ test).reshape((7,7)))
        
#         left = torch.diag_embed(torch.ones(self.state_dim), offset=1).to('cuda')
#         left = left[:-1, :-1]
#         left[[0,7,14,21,28,35,42], [0,7,14,21,28,35,42]] = 1
        
#         for (row, col) in self.wall_idcs:
#             idx = col + row * 7
#             left[:,idx+1] = 0
#             left[idx+1,idx+1] = 1
#         print((left @ test).reshape((7,7)))
        
#         up = torch.diag_embed(torch.ones(self.state_dim), offset=7).to('cuda')
#         up = up[:-7, :-7]
#         up[:7,:7] = torch.diag_embed(torch.ones(7)).to('cuda')
#         print((up @ test).reshape((7,7)))
#         self.actions = [down, up, right, left]
        
#     def forward(self, state, action):
#         # print(state.reshape((1, 7, 7)))
#         # print(action)
#         out = torch.stack([self.actions[a] @ state[i] for i, a in enumerate(action)], dim=0)
#         out = out / torch.sum(out, dim=-1, keepdim=True)
#         # print(out.reshape((1,7,7)))
#         # raise ValueError
#         return out


# predicts the value prefix
# takes in sequence (s_t, hat_s_t+1, ..., hat_s_t+k-1)
# and predicts the k step value prefix for each
class ValuePrefixPredictor(nn.Module):
    def __init__(self, num_variables, codebook_size, embedding_dim, num_values, mlp_hidden_dims):
        super().__init__()
        self.num_variables = num_variables
        self.codebook_size = codebook_size
        self.embedding_dim = embedding_dim
        self.num_values = num_values
        self.mlp_hidden_dims = mlp_hidden_dims
        
        self.latent_embedding = nn.ModuleList([nn.Embedding(codebook_size, embedding_dim) for _ in range(num_variables)])
        # for m in self.latent_embedding:
        #     nn.init.xavier_uniform_(m.weight)
            
        mlp_list = [nn.Linear(embedding_dim*num_variables, self.mlp_hidden_dims[0])]
        for i, dim in enumerate(self.mlp_hidden_dims[1:]):
            mlp_list.extend([
                # nn.BatchNorm1d(self.mlp_hidden_dims[i]), 
                nn.ReLU(), 
                nn.Linear(self.mlp_hidden_dims[i], dim)
            ])
        mlp_list.extend([
            # nn.BatchNorm1d(self.mlp_hidden_dims[-1]),
            nn.ReLU(),
            nn.Linear(self.mlp_hidden_dims[-1], self.num_values),
        ])
        self.mlp = nn.Sequential(*mlp_list)
        
        self.all_idcs = generate_all_combinations(codebook_size, num_variables).to(self.device)
        
        
    def forward(self):
        all_embeds = self.get_all_embeds()
        return self.mlp(all_embeds)

    def get_all_embeds(self):
        self.all_idcs = self.all_idcs.to(self.device)
        return torch.cat([self.latent_embedding[j](self.all_idcs[:,j]) for j in range(self.num_variables)], dim=1)

    @property
    def device(self):
        return next(self.parameters()).device

# combine everythin
class DiscreteNet(nn.Module):
    def __init__(
        self,
        state_prior: StatePrior,
        transition,
        value_prefix_predictor: ValuePrefixPredictor,
        emission,
        kl_balancing_coeff: float,
        l_unroll: int,
        discount_factor: float,
        reward_support,
        disable_recon_loss: bool = True,
        sparsemax: bool = False,
        sparsemax_k: int = 30,
        kl_scaling: float = 1.0,
    ):
        if l_unroll < 1:
            raise ValueError('l_unroll must be at least 1')
        
        super().__init__()
        self.state_prior = state_prior
        self.transition = transition
        self.value_prefix_predictor = value_prefix_predictor
        self.disable_vp = value_prefix_predictor is None
        self.emission = emission
        self.kl_balancing_coeff = kl_balancing_coeff
        self.l_unroll = l_unroll
        self.discount_factor = discount_factor
        self.reward_support = reward_support
        self.disable_recon_loss = disable_recon_loss
        self.sparsemax = sparsemax
        self.sparsemax_k = sparsemax_k
        self.kl_scaling = kl_scaling

        self.discount_array = self.discount_factor ** torch.arange(self.l_unroll)
        
    # def compute_value_prefixes(self, reward_sequence):
    #     if self.discount_array.device != reward_sequence.device:
    #         self.discount_array.to(reward_sequence.device)
    #     reward_sequence = torch.cat([reward_sequence, torch.zeros((reward_sequence.shape[0], self.l_unroll-1), device=reward_sequence.device)], dim=1)
    #     return F.conv1d(reward_sequence[:,None], self.discount_array[None,None,:])[:,0]

    def compute_posterior(self, prior, state_idcs, obs_frame, dropped):
        obs_logits = self.emission(obs_frame, state_idcs)
        
        # posterior = prior.clone()
        # posterior[prior > 0] = posterior[prior > 0].log() + (obs_logits[None][prior > 0,:] * (1-dropped)).sum(dim=-1)
        # posterior[prior > 0] = F.softmax(posterior[prior > 0], dim=-1)
        # print(f"F.log_softmax(obs_logits[None], dim=-1): {F.log_softmax(obs_logits[None], dim=-1)}")
        posterior = prior.clone()
        posterior = posterior * torch.exp((obs_logits[None] * (1-dropped)).sum(dim=-1))
        posterior = posterior / posterior.sum(dim=-1, keepdim=True)

        # print(posterior)
        # print(obs_logits)
        # print(dropped)
        # print((F.log_softmax(obs_logits[None], dim=-1)[prior > 0, :] * (1-dropped)).sum(dim=-1).exp())
        # print(posterior2)
        # print(torch.isclose(posterior, posterior2))
        # print(all(torch.isclose(posterior, posterior2)))
        
        return posterior, obs_logits

    def forward(self, obs_sequence, action_sequence, value_prefix_sequence, dropped):
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
            value_prefix_means = self.value_prefix_predictor()[:,0] # has shape (num_states, )
        else:
            value_prefix_means = None

        # prior
        state_belief = self.state_prior(batch_size)
        
        if self.sparsemax:
            # print(self.state_prior.prior.data)
            state_belief, state_idcs = sparsemax_k(state_belief[0], self.sparsemax_k) 
            #TODO add batch support
            state_belief = state_belief[None]
        else:
            state_belief = F.softmax(state_belief, dim=-1)
            state_idcs = None

        prior_entropy = discrete_entropy(state_belief)

        # unnormalized p(z_t|x_t)
        # print(state_belief)
        posterior_0, obs_logits_0 = self.compute_posterior(state_belief, state_idcs, obs_sequence[:,0], dropped[:,0])
        posterior_entropy = discrete_entropy(posterior_0)        

        # pull posterior and prior closer together
        prior_loss = self.kl_balancing_loss(state_belief, posterior_0)
        state_belief = posterior_0
        
        # dynamics
        #TODO for sparse stuff
        posterior_belief_sequence, posterior_idcs_sequence, prior_entropies, posterior_entropies, obs_logits_sequence, value_prefix_pred, dyn_loss = self.process_sequence(
            action_sequence, 
            dropped, 
            state_belief, 
            state_idcs, 
            value_prefix_means, 
            obs_sequence
        )
        
        obs_logits_sequence = torch.cat([obs_logits_0[:,None], obs_logits_sequence], dim=1)
        
        # compute losses
        recon_loss = self.compute_recon_loss(posterior_belief_sequence, posterior_idcs_sequence, obs_logits_sequence, dropped)
        # print(f"{recon_loss = }")
        if not self.disable_vp:
            value_prefix_loss = self.compute_vp_loss(value_prefix_pred, target_value_prefixes) # TODO if we use longer rollout need to adapt this
        else:
            value_prefix_loss = 0
        # log the losses
        outputs['prior_loss'] = prior_loss
        outputs['value_prefix_loss'] = value_prefix_loss
        outputs['recon_loss'] = recon_loss / dimension
        outputs["dyn_loss"] = self.kl_scaling * dyn_loss / self.l_unroll
        outputs['prior_entropy'] = (prior_entropy + sum(prior_entropies))/(len(prior_entropies) + 1)
        outputs['posterior_entropy'] = (posterior_entropy + sum(posterior_entropies))/(len(posterior_entropies) + 1)
        
        return outputs

    def convert_problogs_to_probs(self, logits, dim):
        if self.sparsemax:
            return sparsemax_k(logits, dim=dim, k=self.sparsemax_k)
        else:
            return F.softmax(logits, dim=dim)

    def process_sequence(self, action_sequence, dropped, state_belief, state_idcs, value_prefix_means, obs_sequence):
        prior_sequences = deque([{"belief": [], "idcs": []} for i in range(self.l_unroll)], maxlen=self.l_unroll)
        value_prefix_loss = 0
        dyn_loss = 0
        value_prefix_pred = []
        posterior_belief_sequence = [state_belief]
        posterior_idcs_sequence = [state_idcs]
        obs_logits_sequence = []
        prior_entropies = []
        posterior_entropies = []
        
        if self.l_unroll > 1:
            raise ValueError("l_unroll > 1 not implemented -> posterior update will not work as intended, also value prefix is not gonna work")
        for t in range(1,action_sequence.shape[1]):
            # extrapolate l_unroll steps, ignore the last action since we can't score the results of that
            state_belief_prior_sequence, state_idcs_prior_sequence = self.k_step_extrapolation(state_belief, state_idcs, action_sequence[:, t-1:-1])
            for i in range(state_belief_prior_sequence.shape[1]):
                prior_sequences[i]['belief'].append(state_belief_prior_sequence[:,i])
                prior_sequences[i]['idcs'].append(state_idcs_prior_sequence[i])
            
            if not self.disable_vp:
                # predict value prefixes
                value_prefix_pred.append(state_belief @ value_prefix_means[state_idcs])
                
            # get the priors for the next state
            obj = prior_sequences.popleft()
            priors = torch.stack(obj['belief'], dim=1)
            prior_idcs = obj['idcs']
            prior_sequences.append({"belief": [], "idcs": []})
            
            # get the posterior for the next state
            state_belief_posterior, obs_logits = self.compute_posterior(priors[:,-1], prior_idcs[-1], obs_sequence[:,t], dropped[:,t])
            # compute the dynamics loss
            dyn_loss = dyn_loss + sum(self.kl_balancing_loss(priors[:,i], state_belief_posterior) for i in range(priors.shape[1]))
            
            # set belief to the posterior
            state_belief = state_belief_posterior
            state_idcs = prior_idcs[-1]

            posterior_belief_sequence.append(state_belief_posterior)
            posterior_idcs_sequence.append(prior_idcs[-1])
            obs_logits_sequence.append(obs_logits)
            prior_entropies.append(discrete_entropy(priors[:,-1]))
            posterior_entropies.append(discrete_entropy(state_belief_posterior))
            
        obs_logits_sequence = torch.stack(obs_logits_sequence, dim=1)
        posterior_belief_sequence = torch.stack(posterior_belief_sequence, dim=1)
        posterior_idcs_sequence = posterior_idcs_sequence # TODO add native batch suppprt
        if not self.disable_vp:
            value_prefix_pred = torch.stack(value_prefix_pred, dim=1)
        else:
            value_prefix_pred = None
        
        
        return posterior_belief_sequence, posterior_idcs_sequence, prior_entropies, posterior_entropies, obs_logits_sequence, value_prefix_pred, dyn_loss

    def compute_recon_loss(self, posterior_belief_sequence, posterior_idcs_sequence, obs_logits_sequence, dropped):
        # posterior_belief_sequence has shape (batch, seq_len, num_states)
        # obs_logits_sequence has shape (num_active_state, seq_len, num_views)
        # dropped has shape (batch, seq_len, num_views)
        recon_loss = 0
        if not self.disable_recon_loss:
            recon_loss = (-(1-dropped)[:,:,None,:] * posterior_belief_sequence[:,:,:,None] * einops.rearrange(obs_logits_sequence,"num_states seq views -> seq num_states views")[None]).sum(dim=[2,-1]).mean()
        return recon_loss

    def compute_vp_loss(self, value_prefix_pred, target_value_prefixes):
        return F.mse_loss(value_prefix_pred, target_value_prefixes[:,1:])
        

    def kl_balancing_loss(self, prior, posterior):
        return (self.kl_balancing_coeff * discrete_kl(posterior.detach(), prior) + (1 - self.kl_balancing_coeff) * discrete_kl(posterior, prior.detach()))

    def k_step_extrapolation(self, state_belief, state_idcs, action_sequence, k=None):
        if k is None:
            k = self.l_unroll
        state_belief_prior_sequence = []
        state_idcs_prior_sequence = []
        for t in range(min(action_sequence.shape[1], k)):
            state_belief, state_idcs = self.transition(state_belief, state_idcs, action_sequence[:, t])
            state_belief = state_belief / state_belief.sum(dim=-1, keepdim=True)
            state_belief_prior_sequence.append(state_belief)
            state_idcs_prior_sequence.append(state_idcs)
        # print(state_belief_prior_sequence)
        # print(state_idcs_prior_sequence)
        return torch.stack(state_belief_prior_sequence, dim=1), torch.stack(state_idcs_prior_sequence, dim=0)

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
        # factorized_transition,
        transition_mode,
        reward_support,
        disable_recon_loss = False,
        sparsemax = False,
        sparsemax_k = 30,
        attention_batch_size = -1,
        disable_vp = False,
        action_layer_dims = None,
        kl_scaling=1.0,
    ):
        super().__init__()
        self.save_hyperparameters()
        
        #TODO factorize num_variables etc. to be top-level parameters
        self.prior = StatePrior(emission_kwargs['num_variables'], emission_kwargs['codebook_size'], device)
        if transition_mode == 'factorized':
            self.transition = ActionConditionedFactorizedTransition(state_dim, num_actions, layer_dims=action_layer_dims, batch_size=attention_batch_size)
        elif transition_mode == 'matrix':
            self.transition = ActionConditionedTransition(state_dim, num_actions, device)
        elif transition_mode == 'mlp':
            self.transition = ActionConditionedMLPTransition(state_dim, num_actions)
        else:
            raise ValueError(f'Unknown transition mode: {transition_mode}')
        if not disable_vp:
            self.value_prefix_predictor = ValuePrefixPredictor(emission_kwargs['num_variables'], emission_kwargs['codebook_size'], emission_kwargs['embedding_dim'], **vp_kwargs)
        else:
            self.value_prefix_predictor = None
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
        )
    
    def forward(self, obs, actions, value_prefixes, dropped):
        return self.network(obs, actions, value_prefixes, dropped)
    
    def training_step(self, batch, batch_idx):
        outputs = self(*batch)
        total_loss = sum(list(val for (key, val) in outputs.items() if not key.endswith('entropy')))
        for key, value in outputs.items():
            self.log(f"Training/{key}", value)
        self.log(f"Training/total_loss", total_loss)
        # for key, value in outputs.items():
        #     print(f"{key}: {value}")
        
        return total_loss
    
    def validation_step(self, batch, batch_idx):
        outputs = self(*batch)
        total_loss = sum(list(val for (key, val) in outputs.items() if not key.endswith('entropy')))
        for key, value in outputs.items():
            self.log(f"Validation/{key}", value)
        self.log(f"Validation/total_loss", total_loss)
        
        return total_loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.network.parameters(), lr=self.hparams.learning_rate, weight_decay=self.hparams.weight_decay)
        # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)
        return optimizer
            
    def backward(self, loss, optimizer, *arg):
        loss.backward(retain_graph=True)


class MLPDecoder(nn.Module):
    def __init__(self, latent_dim, num_vars, output_channels, width=7) -> None:
        super().__init__()
        
        self.net = nn.Sequential(
            layers.Rearrange('b n d -> b (n d)'),
            nn.Linear(latent_dim*num_vars, 256),
            nn.GELU(),
            nn.Linear(256, 256),
            nn.GELU(),
            nn.Linear(256, output_channels * (width ** 2)),
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


class Decoder(nn.Module):

    def __init__(self, latent_dim=32, num_vars=32, output_channels=3):
        super().__init__()
        # dreamer
        self.net = nn.Sequential(
            layers.Rearrange('b n d -> b (n d)'),
            nn.Linear(latent_dim*num_vars, 1024),
            layers.Rearrange('b (d h w) -> b d h w', h=4, w=4),
            nn.ConvTranspose2d(64, 32, (2,2), (1,1)),
            nn.ELU(alpha=1.0),
            nn.ConvTranspose2d(32, 16, (2,2), (1,1)),
            nn.ELU(alpha=1.0),
            # nn.ConvTranspose2d(16, output_channels, (2,2), (1,1)),
            nn.ConvTranspose2d(16, 16, (2,2), (1,1)),
            layers.Rearrange('b d h w -> b (d h w)'),
            nn.Linear(16*49, output_channels*49),
            layers.Rearrange('b (d h w) -> b d h w', h=7, w=7),
        )

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
        return self.net(x)

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
    ):
        super().__init__()
        self.num_input_channels = num_input_channels
        self.embedding_dim = embedding_dim
        self.codebook_size = codebook_size
        self.num_variables = num_variables
        self.sparse = sparse
        # self.precomputed_states = dict()
        
        if mlp:
            self.decoder = MLPDecoder(embedding_dim, num_variables, output_channels=num_input_channels, width=7)
        else:
            self.decoder = Decoder(latent_dim=embedding_dim, num_vars=num_variables, output_channels=num_input_channels)
        self.latent_embedding = nn.Parameter(torch.zeros(num_variables, codebook_size, embedding_dim))
        nn.init.normal_(self.latent_embedding)
        
        # for m in self.latent_embedding:
        #     nn.init.xavier_uniform_(m.weight)
        print(self)
        
        self.all_idcs = generate_all_combinations(codebook_size, num_variables)
        self.batch_size = self.all_idcs.shape[0] # TODO make this an arg

    def forward(self, x, active_states: Optional[Tensor]=None):
        if self.sparse:
            emission_means = self.get_emission_means_sparse(active_states, x.device)
            # print(emission_means.shape)
            obs_logits = self.compute_obs_logits_sparse(x, emission_means)
        else:
            # assuming a diagonal gaussian with unit variance
            emission_means = self.get_emission_means()
            obs_logits = self.compute_obs_logits(x, emission_means)
        return obs_logits

    def get_emission_means_sparse(self, active_states, device=None):
        # TODO add batch support
        embeds = []
        # print('ho')
        # print(active_states)
        states = []
        for i in range(len(active_states)):
            state_idx = active_states[i]
            # if state_idx not in self.precomputed_states:
            #     self.precomputed_states[state_idx] = F.one_hot(convert_state_idx_to_bit_vector(state_idx, self.num_variables, device), num_classes=2).float()
            # states.append(self.precomputed_states[state_idx])
            states.append(F.one_hot(convert_state_idx_to_bit_vector(state_idx, self.num_variables, device), num_classes=2).float())
                # self.precomputed_states[state_idx] = convert_state_idx_to_bit_vector(state_idx, self.num_variables, device)
        states = torch.stack(states, dim=0)
        embeds = torch.einsum("ktc,tce->kte", states, self.latent_embedding)
        return self.decoder(embeds)

    def get_emission_means(self):
        emission_probs = []
        for i in range(math.ceil(len(self.all_idcs)//self.batch_size)):
            idcs = self.all_idcs[i*self.batch_size:(i+1)*self.batch_size].to(self.device)
            z = torch.stack([self.latent_embedding[j](idcs[:,j]) for j in range(self.num_variables)], dim=1)
            x_dist = self.decoder(z)
            emission_probs.append(x_dist)
        emission_probs = torch.cat(emission_probs, dim=0)
        return torch.reshape(emission_probs, (*(self.codebook_size,)*self.num_variables, *x_dist.shape[1:]))
        
    def compute_obs_logits(self, x, emission_means):
        #TODO separate channels and views rather than treating them interchangably?
        output = - ((emission_means[None] - x[(slice(None),) + (None,)*self.num_variables]) ** 2).sum(dim=[-2,-1]) / 2 #- math.log(2*math.pi) / 2 * emission_means.shape[-2] * emission_means.shape[-1]
        output = einops.rearrange(output, 'batch ... num_views -> batch num_views ...')
        return output

    def compute_obs_logits_sparse(self, x, emission_means):
        # TODO add batch support
        #TODO separate channels and views rather than treating them interchangably?
        output = - ((emission_means - x[None,0]) ** 2).sum(dim=[-2,-1]) / 2 #- math.log(2*math.pi) / 2 * emission_means.shape[-2] * emission_means.shape[-1]
        return output
    
    
    ## hardcoded for numvars = 1, codebooksize 49
    # def compute_obs_logits(self, x, emission_means):
    #     #TODO separate channels and views rather than treating them interchangably?
    #     output = einops.rearrange(torch.zeros_like(x), '... views h w -> ... views (h w)') - 10 ** 10
    #     output[torch.arange(len(x)),...,torch.argmin(einops.rearrange(x, '... views h w -> ... (views h w)'), dim=-1)] = 0
    #     return output
    
    
    @torch.no_grad()
    def decode_only(self, latent_dist, idcs=None):
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
            emission_means = torch.stack([self.get_emission_means_sparse(idx, device=latent_dist.device) for idx in idcs], dim=0)
            mean_prediction = (latent_dist[...,None,None,None] * emission_means).sum(dim=1)
        else:
            for i, idcs in enumerate(itertools.product(range(self.codebook_size), repeat=self.num_variables)):
                probs = latent_dist[:, i]
                idcs = torch.tensor(list(idcs), dtype=torch.long)[None].to(self.device)
                # embed
                embedded_latents = torch.stack([self.latent_embedding[j](idcs[:,j]) for j in range(self.num_variables)], dim=1)
                # decode
                
                x_hat = probs[:,None,None,None] * self.decoder(embedded_latents)
                if mean_prediction is None:
                    mean_prediction = x_hat
                else:
                    mean_prediction += x_hat

        return mean_prediction

    @property
    def device(self):
        return next(self.parameters()).device
    
def generate_all_combinations(codebook_size, num_variables):
        all_idcs = []
        for i, idcs in enumerate(itertools.product(range(codebook_size), repeat=num_variables)):
            all_idcs.append(list(idcs))
        return torch.tensor(all_idcs, dtype=torch.long)
 
    
def test_generate_all_combinations():
    codebook_size = 3
    num_variables = 2
    
    all_combos = generate_all_combinations(codebook_size, num_variables)
    target = torch.tensor([[0,0],[0,1],[0,2],[1,0],[1,1],[1,2],[2,0],[2,1],[2,2]], dtype=torch.long)
    assert torch.equal(all_combos, target)


test_generate_all_combinations()

    
