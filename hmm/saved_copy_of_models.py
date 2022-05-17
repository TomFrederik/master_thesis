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


from utils import discrete_kl

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
    def __init__(self, state_dim, device):
        super().__init__()
        self.state_dim = state_dim

        # init prior to uniform distribution
        self.prior = nn.Parameter(torch.zeros(self.state_dim, device=device))
        
    def forward(self, batch_size):
        # print(f"prior: {self.prior}")
        return einops.repeat(torch.softmax(self.prior, dim=0), 'dim -> batch_size dim', batch_size=batch_size)

    def to(self, device):
        self.prior = self.prior.to(device)
    
    @property
    def device(self):
        return self.prior.device

# models the transition distribution of the latent space conditioned on the action
# instantiates |A| transition matrices
class ActionConditionedTransition(nn.Module):
    def __init__(self, state_dim, num_actions, device, factorized=False):
        super().__init__()
        self.state_dim = state_dim
        self.num_actions = num_actions
        self.factorized = factorized
        
        if factorized:
            raise NotImplementedError
        else:
            self.matrices = nn.Parameter(torch.zeros((self.num_actions, self.state_dim, self.state_dim), device=device))
    
    def forward(self, state, action):
        if self.factorized:
            raise NotImplementedError
        else:
            matrix = self.matrices[action]
            return torch.einsum('bi, bij -> bj', state, torch.softmax(matrix, dim=-1))
    
    def to(self, device):
        self.matrices = self.matrices.to(device)

class ActionConditionedMLPTransition(nn.Module):
    def __init__(self, state_dim, num_actions, hidden_dim=128):
        super().__init__()
        self.state_dim = state_dim
        self.num_actions = num_actions
        
        self.action_emb = nn.Embedding(num_actions, hidden_dim)
        self.emb = nn.Linear(state_dim, hidden_dim)
        self.linear1 = nn.Linear(2*hidden_dim, 2*hidden_dim)
        self.linear2 = nn.Linear(2*hidden_dim, state_dim)

        # init
        self._init_params()
        
    def _init_params(self):
        torch.nn.init.kaiming_uniform_(self.linear1.weight, mode='fan_out')
        torch.nn.init.zeros_(self.linear1.bias)
        torch.nn.init.zeros_(self.linear2.bias)
    
    def forward(self, state, action):
        state = self.emb(state)
        action = self.action_emb(action)
        
        return torch.softmax(self.linear2(F.relu(self.linear1(torch.cat((state, action), dim=-1)))), dim=-1)

# predicts the value prefix
# takes in sequence (s_t, hat_s_t+1, ..., hat_s_t+k-1)
# and predicts the k step value prefix for each
class ValuePrefixPredictor(nn.Module):
    def __init__(self, state_dim, num_values, lstm_hidden_dim, num_lstm_layers, mlp_hidden_dims):
        super().__init__()
        self.state_dim = state_dim
        self.num_values = num_values
        self.lstm_hidden_dim = lstm_hidden_dim
        self.num_lstm_layers = num_lstm_layers
        self.mlp_hidden_dims = mlp_hidden_dims
        
        self.lstm = nn.LSTM(self.state_dim, self.lstm_hidden_dim, self.num_lstm_layers, batch_first=True)
        
        mlp_list = [nn.Linear(self.lstm_hidden_dim, self.mlp_hidden_dims[0])]
        for i, dim in enumerate(self.mlp_hidden_dims[1:]):
            mlp_list.extend([
                nn.BatchNorm1d(self.mlp_hidden_dims[i]), 
                nn.ReLU(inplace=True), 
                nn.Linear(self.mlp_hidden_dims[i], dim)
            ])
        mlp_list.extend([
            nn.BatchNorm1d(self.mlp_hidden_dims[-1]),
            nn.ReLU(),
            nn.Linear(self.mlp_hidden_dims[-1], self.num_values),
        ])
        self.mlp = nn.Sequential(*mlp_list)
        
        self.network = nn.Sequential(
            self.lstm,
            layers.Rearrange('batch seq dim -> (batch seq) dim'),
            nn.BatchNorm1d(self.lstm_hidden_dim),
            nn.ReLU(inplace=True),
            self.mlp,
        )
        
    def forward(self, state_sequence):
        batch, seq, dim = state_sequence.shape
        out = self.network(state_sequence)
        return einops.rearrange(out, '(batch seq) dim -> batch seq dim', batch=batch, seq=seq)

# combine everythin
class DiscreteNet(nn.Module):
    def __init__(
        self,
        state_prior,
        transition,
        # value_prefix_predictor,
        emission,
        kl_balancing_coeff,
        l_unroll,
        discount_factor,
        # reward_support,
    ):
        super().__init__()
        self.state_prior = state_prior
        self.transition = transition
        # self.value_prefix_predictor = value_prefix_predictor
        self.emission = emission
        self.kl_balancing_coeff = kl_balancing_coeff
        self.l_unroll = l_unroll
        self.discount_factor = discount_factor
        # self.reward_support = reward_support

        self.discount_array = self.discount_factor ** torch.arange(self.l_unroll)
        
    # def compute_value_prefixes(self, reward_sequence):
    #     if self.discount_array.device != reward_sequence.device:
    #         self.discount_array.to(reward_sequence.device)
    #     reward_sequence = torch.cat([reward_sequence, torch.zeros((reward_sequence.shape[0], self.l_unroll-1), device=reward_sequence.device)], dim=1)
    #     return F.conv1d(reward_sequence[:,None], self.discount_array[None,None,:])[:,0]

    def forward(self, obs_sequence, action_sequence, reward_sequence, dropped, player_pos):
        outputs = dict()
        
        batch_size, seq_len, *_ = obs_sequence.shape
        # # compute target value prefixes
        # print(f"{reward_sequence.shape = }")
        # target_value_prefixes = self.compute_value_prefixes(reward_sequence)
        # print(f"{target_value_prefixes.shape = }")
        
        # # convert them to classes
        # transformed_target_value_prefixes = self.scalar_transform(target_value_prefixes)
        # target_value_prefixes_phi = self.reward_phi(transformed_target_value_prefixes)
        
        # emission model to get emission probs for all possible latent states
        # obs sequence has shape (batch, seq, num_views, 7, 7)
        batch, seq, views = obs_sequence.shape[:3]
        obs_logits = self.emission(einops.rearrange(obs_sequence, 'batch seq views ... -> (batch seq) views ...'))
        obs_logits = einops.rearrange(obs_logits, '(batch seq) views ... -> batch seq views (...)', batch=batch, seq=seq, views=views) # flatten over latent states
            
        # prior
        state_belief = self.state_prior(batch_size)
        # compute recon loss at step 0        
        # only count loss for non-dropped views
        recon_loss = obs_logits[:,0] * (1-dropped)[:,0,:,None]
        recon_loss = recon_loss.sum(dim=1) # sum logits over views
        recon_loss = recon_loss + state_belief.log() # TODO: Is this right?
        recon_loss = torch.logsumexp(recon_loss, dim=-1) # LSE over latents

        # compute posterior
        # don't use the updates from the dropped frame
        # sum over views
        posterior_0 = state_belief.log() + (obs_logits[:,0] * (1-dropped)[:,0,:,None]).sum(dim=1) # TODO: re-use computation from above
        posterior_0 = torch.nn.functional.softmax(posterior_0, dim=1)
        # pull posterior and prior closer together
        prior_loss = self.kl_balancing_loss(state_belief, posterior_0)
        state_belief = posterior_0
        outputs['prior_loss'] = prior_loss
        # dynamics
        prior_sequences = deque([[] for i in range(self.l_unroll)], maxlen=self.l_unroll)
        value_prefix_loss = 0
        dyn_loss = 0
        # print(f"{action_sequence.shape = }")
        if self.l_unroll > 0:
            for t in range(1,seq_len):
                # extrapolate l_unroll steps, ignore the last action since we can't score the results of that
                state_belief_prior_sequence = self.k_step_extrapolation(state_belief, action_sequence[:, t-1:-1])
                for i in range(state_belief_prior_sequence.shape[1]):
                    print('i', i)
                    prior_sequences[i].append(state_belief_prior_sequence[:,i])
                
                # TODO
                # predict value prefixes
                # value_prefix_logits = self.value_prefix_predictor(torch.cat([state_belief, state_belief_prior_sequence], dim=1))
                # print(f"{value_prefix_logits.shape = }")
                # score them against the actual value prefixes
                # value_prefix_loss += F.cross_entropy(value_prefix_logits, target_value_prefixes_phi[:,t])
                
                # get the priors for the next state
                priors = torch.stack(prior_sequences.popleft(), dim=1)
                prior_sequences.append([])
                print(f"{priors.shape = }")
                print(f"{obs_logits.shape = }")
                # compute p(x|z)p(z) and then the recon loss
                recon_loss = recon_loss + torch.logsumexp((priors[:,-1].log() + (obs_logits[:,t] * (1-dropped)[:,t,:,None]).sum(dim=1)), dim=-1)
                
                # get the posterior for the next state
                state_belief_posterior = priors[:,-1].log() + (obs_logits[:,t]*(1-dropped)[:,t,:,None]).sum(dim=1)
                state_belief_posterior = torch.nn.functional.softmax(state_belief_posterior, dim=1)
                print(f"{state_belief_posterior.shape = }")
                
                # compute the dynamics loss
                print('t', t, sum(self.kl_balancing_loss(priors[:,i], state_belief_posterior) for i in range(priors.shape[1])))
                dyn_loss = dyn_loss + sum(self.kl_balancing_loss(priors[:,i], state_belief_posterior) for i in range(priors.shape[1]))
                
                # set belief to the posterior
                state_belief = state_belief_posterior
        
        # take mean of recon loss
        outputs['recon_loss'] = -recon_loss.mean()
        
        # take mean of value prefix loss
        outputs['value_prefix_loss'] = value_prefix_loss
        
        # take mean of dyn loss?
        outputs["dyn_loss"] = dyn_loss
        
        return outputs

    def kl_balancing_loss(self, prior, posterior):
        return (self.kl_balancing_coeff * discrete_kl(posterior.detach(), prior) + (1 - self.kl_balancing_coeff) * discrete_kl(posterior, prior.detach()))

    def k_step_extrapolation(self, state_belief, action_sequence, k=None):
        if k is None:
            k = self.l_unroll
        state_belief_prior_sequence = []
        for t in range(min(action_sequence.shape[1], k)):
            state_belief = self.transition(state_belief, action_sequence[:, t])
            state_belief = state_belief / state_belief.sum(dim=-1, keepdim=True)
            state_belief_prior_sequence.append(state_belief)
        return torch.stack(state_belief_prior_sequence, dim=1)

    ##########################################################
    # The following functions are from the EfficientZero paper
    # https://github.com/YeWR/EfficientZero/blob/a0c094818d750237d5aa14263a65a1e1e4f2bbcb/core/config.py
    ##########################################################
    # def scalar_transform(self, x):
    #     """ Reference from MuZero: Appendix F => Network Architecture
    #     & Appendix A : Proposition A.2 in https://arxiv.org/pdf/1805.11593.pdf (Page-11)
    #     """
    #     epsilon = 0.001
    #     sign = torch.ones(x.shape).float().to(x.device)
    #     sign[x < 0] = -1.0
    #     output = sign * (torch.sqrt(torch.abs(x) + 1) - 1 + epsilon * x)
    #     return output

    # def reward_phi(self, x):
    #     return self._phi(x, self.reward_support.min, self.reward_support.max, self.reward_support.size)

    # def _phi(self, x, min, max, set_size: int):
    #     x.clamp_(min, max)
    #     x_low = x.floor()
    #     x_high = x.ceil()
    #     p_high = x - x_low
    #     p_low = 1 - p_high

    #     target = torch.zeros(x.shape[0], x.shape[1], set_size).to(x.device)
    #     x_high_idx, x_low_idx = x_high - min, x_low - min
    #     target.scatter_(2, x_high_idx.long().unsqueeze(-1), p_high.unsqueeze(-1))
    #     target.scatter_(2, x_low_idx.long().unsqueeze(-1), p_low.unsqueeze(-1))
    #     return target
    ##########################################################
    ##########################################################

class LightningNet(pl.LightningModule):
    def __init__(
        self,
        state_dim,
        num_actions,
        # vp_kwargs,
        emission_kwargs,
        kl_balancing_coeff,
        l_unroll,
        discount_factor,
        learning_rate,
        device,
        # factorized_transition,
        mlp_transition,
        # reward_support,
    ):
        super().__init__()
        self.save_hyperparameters()
        
        self.prior = StatePrior(state_dim, device)
        if mlp_transition:
            self.transition = ActionConditionedMLPTransition(state_dim, num_actions)
        else:
            self.transition = ActionConditionedTransition(state_dim, num_actions, device)
        # self.value_prefix_predictor = ValuePrefixPredictor(state_dim, **vp_kwargs)
        self.emission = EmissionModel(**emission_kwargs) #TODO
        self.network = DiscreteNet(
            self.prior,
            self.transition,
            # self.value_prefix_predictor,
            self.emission,
            kl_balancing_coeff,
            l_unroll,
            discount_factor,
            # reward_support,
        )
    
    def forward(self, obs, actions, rewards, dropped, player_pos):
        return self.network(obs, actions, rewards, dropped, player_pos)
    
    def training_step(self, batch, batch_idx):
        outputs = self(*batch)
        
        for key, value in outputs.items():
            self.log(f"Training/{key}", value)
        self.log(f"Training/total_loss", sum(list(outputs.values())))
        # for key, value in outputs.items():
        #     print(f"{key}: {value}")
        
        return sum(list(outputs.values()))
    
    def validation_step(self, batch, batch_idx):
        outputs = self(*batch)
        
        for key, value in outputs.items():
            self.log(f"Validation/{key}", value)
        self.log(f"Validation/total_loss", sum(list(outputs.values())))
        
        return sum(list(outputs.values()))
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.network.parameters(), lr=self.hparams.learning_rate)
        # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)
        return optimizer
            
    def backward(self, loss, optimizer, *arg):
        loss.backward(retain_graph=True)


class TotalQuantizer(nn.Module):
    def __init__(self, num_variables, codebook_size, embedding_dim, entropy_scale):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.codebook_size = codebook_size
        self.num_variables = num_variables

        self.entropy_scale = entropy_scale

        self.embeds = nn.ModuleList([nn.Embedding(codebook_size, embedding_dim) for _ in range(self.num_variables)])

    def forward(self, logits):
        logging.warning("TotalQuantizer.forward() is deprecated. Don't use!")
        logits = einops.rearrange(logits, 'b (num_variables codebook_size) -> b num_variables codebook_size', codebook_size=self.codebook_size, num_variables=self.num_variables)
        latent_dist = F.softmax(logits, dim=2)
        latent_loss = self.entropy_scale * torch.sum(latent_dist * torch.log(latent_dist + 1e-10), dim=[1,2])

        return latent_dist, latent_loss

    def forward_one_hot(self, logits):
        raise NotImplementedError
        logits = einops.rearrange(logits, 'b (num_variables codebook_size) -> b num_variables codebook_size', codebook_size=self.codebook_size, num_variables=self.num_variables)

        probs = torch.softmax(logits, dim=2)
        one_hot = F.gumbel_softmax(logits, tau=self.temperature, dim=2, hard=True)
        return one_hot, probs

    def embed_idcs(self, idcs):
        idcs = idcs.to(self.embeds[0].weight.device)
        z = torch.stack([self.embeds[i](idcs[:,i]) for i in range(self.num_variables)], dim=1)
        return z
        

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
        # self.net = nn.Sequential(
        #     layers.Rearrange('b n d -> b (n d)'),
        #     nn.Linear(latent_dim*num_vars, 1024),
        #     layers.Rearrange('b (d h w) -> b d h w', h=4, w=4),
        #     nn.ConvTranspose2d(64, 32, (2,2), (1,1)),
        #     nn.ELU(alpha=1.0),
        #     nn.ConvTranspose2d(32, 16, (2,2), (1,1)),
        #     nn.ELU(alpha=1.0),
        #     nn.ConvTranspose2d(16, output_channels, (2,2), (1,1)),
        # )

        # ours
        self.net = nn.Sequential(
            layers.Rearrange('b n d -> b (n d)'),
            nn.Linear(latent_dim*num_vars, 2048),
            layers.Rearrange('b (d h w) -> b d h w', h=4, w=4),
            nn.ConvTranspose2d(128, 64, (2,2), (1,1)),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, (2,2), (1,1)),
            nn.ReLU(),
            nn.ConvTranspose2d(32, output_channels, (2,2), (1,1)),
        )
        
        # init
        torch.nn.init.kaiming_uniform_(self.net[3].weight, mode='fan_out')
        torch.nn.init.kaiming_uniform_(self.net[5].weight, mode='fan_out')
        torch.nn.init.zeros_(self.net[1].bias)
        torch.nn.init.zeros_(self.net[3].bias)
        torch.nn.init.zeros_(self.net[5].bias)
        torch.nn.init.zeros_(self.net[7].bias)
        
        
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
    ):
        super().__init__()
        self.num_input_channels = num_input_channels
        self.embedding_dim = embedding_dim
        self.codebook_size = codebook_size
        self.num_variables = num_variables
        
        if mlp:
            self.decoder = MLPDecoder(embedding_dim, num_variables, output_channels=num_input_channels, width=7)
        else:
            self.decoder = Decoder(latent_dim=embedding_dim, num_vars=num_variables, output_channels=num_input_channels)
        self.quantizer = TotalQuantizer(num_variables=num_variables, codebook_size=codebook_size, embedding_dim=embedding_dim, entropy_scale=1)
        
        self.all_idcs = generate_all_combinations(codebook_size, num_variables)
        self.batch_size = self.all_idcs.shape[0] # TODO make this an arg

    def forward(self, x):
        # assuming a diagonal gaussian with unit variance
        emission_means = self.get_emission_means()
        obs_logits = self.compute_obs_logits(x, emission_means)
        return obs_logits
        
    def get_emission_means(self):
        emission_probs = None
        for i in range(math.ceil(len(self.all_idcs)//self.batch_size)):
            z = self.quantizer.embed_idcs(self.all_idcs[i*self.batch_size:(i+1)*self.batch_size])
            x_dist = self.decoder(z)
            if emission_probs is None:
                emission_probs = torch.empty((self.codebook_size**self.num_variables, *x_dist.shape[1:]), device=x_dist.device)
            emission_probs[i*self.batch_size:(i+1)*self.batch_size] = x_dist
            
        return torch.reshape(emission_probs, (*(self.codebook_size,)*self.num_variables, *x_dist.shape[1:]))
        
    def compute_obs_logits(self, x, emission_means):
        #TODO separate channels and views rather than treating them interchangably?
        output = - ((emission_means[None] - x[(slice(None),) + (None,)*self.num_variables]) ** 2).sum(dim=[-2,-1]) / 2 - math.log(2*math.pi) / 2 * emission_means.shape[-2] * emission_means.shape[-1]
        output = einops.rearrange(output, 'batch ... num_views -> batch num_views ...')
        return output
    
    @torch.no_grad()
    def decode_only(self, latent_dist):
        """

        :param latent_dist: shape (batch_size, codebook_size ** num_vars)
        :type latent_dist: torch.Tensor
        :return: x_hat: decoded z
        :rtype: torch.Tensor
        """
        mean_prediction = None

        # TODO make this more efficient using batching
        # compute expected value
        for i, idcs in enumerate(itertools.product(range(self.codebook_size), repeat=self.num_variables)):
            probs = latent_dist[:, i]
            idcs = torch.tensor(list(idcs), dtype=torch.long)[None]
            # embed
            embedded_latents = self.quantizer.embed_idcs(idcs)
            # decode
            
            x_hat = probs[:,None,None,None] * self.decoder(embedded_latents)
            if mean_prediction is None:
                mean_prediction = x_hat
            else:
                mean_prediction += x_hat

        return mean_prediction


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

    