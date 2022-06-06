import itertools
from time import time
from typing import List, Optional, Tuple, Union

import einops
import einops.layers.torch as layers
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F

from commons import Tensor
from sparsemax_k import sparsemax_k, BitConverter
from state_prior import StatePrior
from transition_models import FactorizedTransition
from utils import discrete_entropy, generate_all_combinations, kl_balancing_loss
from value_prefix import ValuePrefixPredictor


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
        self.state_prior: StatePrior = state_prior
        self.transition: FactorizedTransition = transition
        self.value_prefix_predictor: ValuePrefixPredictor = value_prefix_predictor
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
        obs_probs = F.softmax(obs_logits, dim=0) # (num_states, num_views)
        posterior = prior
        for view in range(obs_probs.shape[1]):
            posterior = posterior * obs_probs[:,view][None,:] * (1- dropped)[:,view]
        posterior = posterior / posterior.sum(dim=-1, keepdim=True)

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
            #TODO add batch support
            state_belief, state_bit_vecs = sparsemax_k(state_belief[0], self.sparsemax_k, self.transition.bitconverter) 
            state_belief = state_belief[None]
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
        prior_entropy = discrete_entropy(state_belief)
        posterior_entropy = discrete_entropy(posterior_0)        

        # pull posterior and prior closer together
        prior_loss = kl_balancing_loss(self.kl_balancing_coeff, state_belief, posterior_0)
        state_belief = posterior_0
        
        # dynamics
        posterior_belief_sequence, posterior_bit_vec_sequence, prior_entropies, posterior_entropies, obs_logits_sequence, value_prefix_pred, dyn_loss = self.process_sequence(
            action_sequence, 
            dropped, 
            posterior_0, 
            state_bit_vecs, 
            value_prefix_means, 
            obs_sequence
        )
        
        obs_logits_sequence = torch.cat([obs_logits_0[:,None], obs_logits_sequence], dim=1)

        # compute losses
        recon_loss = self.compute_recon_loss(posterior_belief_sequence, posterior_bit_vec_sequence, obs_logits_sequence, dropped)
        
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

    def process_sequence(self, action_sequence, dropped, state_belief, state_bit_vecs, value_prefix_means, obs_sequence):
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
                if not self.sparsemax:
                    value_prefix_pred.append(posterior_belief_sequence[-1] @ value_prefix_means)
                else:
                    idcs = self.bitconverter.bitvec_to_idx(posterior_bit_vecs_sequence[-1])
                    value_prefix_pred.append(posterior_belief_sequence[-1] @ value_prefix_means[idcs])
            
            # get the priors for the next state
            prior, state_bit_vecs = self.transition(posterior_belief_sequence[-1], action_sequence[:,t-1], posterior_bit_vecs_sequence[-1])
            
            # get the posterior for the next state
            state_belief_posterior, obs_logits = self.compute_posterior(prior, state_bit_vecs, obs_sequence[:,t], dropped[:,t])
            
            # compute the dynamics loss
            dyn_loss = dyn_loss + kl_balancing_loss(self.kl_balancing_coeff, prior, state_belief_posterior)

            # log            
            posterior_belief_sequence.append(state_belief_posterior)
            posterior_bit_vecs_sequence.append(state_bit_vecs)
            obs_logits_sequence.append(obs_logits)
            prior_entropies.append(discrete_entropy(prior))
            posterior_entropies.append(discrete_entropy(state_belief_posterior))
        
        
        obs_logits_sequence = torch.stack(obs_logits_sequence, dim=1)
        posterior_belief_sequence = torch.stack(posterior_belief_sequence, dim=1)
        if not self.disable_vp:
            value_prefix_pred = torch.stack(value_prefix_pred, dim=1)
        else:
            value_prefix_pred = None
        
        return posterior_belief_sequence, posterior_bit_vecs_sequence, prior_entropies, posterior_entropies, obs_logits_sequence, value_prefix_pred, dyn_loss

    def compute_recon_loss(self, posterior_belief_sequence, posterior_bit_vec_sequence, obs_logits_sequence, dropped):
        # posterior_belief_sequence has shape (batch, seq_len, num_states)
        # obs_logits_sequence has shape (num_active_state, seq_len, num_views)
        # dropped has shape (batch, seq_len, num_views)
        recon_loss = 0
        if not self.disable_recon_loss:
            recon_loss = (-(1-dropped)[:,:,None,:] * posterior_belief_sequence[:,:,:,None] * einops.rearrange(obs_logits_sequence,"num_states seq views -> seq num_states views")[None]).sum(dim=[1,2,3]).mean()
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
            state_bit_vec_sequence = torch.stack(state_bit_vec_sequence, dim=0)
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
    ):
        super().__init__()
        self.save_hyperparameters()
        
        #TODO factorize num_variables etc. to be top-level parameters
        self.prior = StatePrior(emission_kwargs['num_variables'], emission_kwargs['codebook_size'], device)
        self.transition = FactorizedTransition(emission_kwargs['num_variables'], emission_kwargs['codebook_size'], num_actions, layer_dims=action_layer_dims, sparse=sparsemax)
        
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
    
    def forward(self, obs, actions, value_prefixes, terms, dropped):
        return self.network(obs, actions, value_prefixes, dropped)
    
    def training_step(self, batch, batch_idx):
        outputs = self(*batch)
        total_loss = sum(list(val for (key, val) in outputs.items() if not key.endswith('entropy')))
        for key, value in outputs.items():
            self.log(f"Training/{key}", value)
        self.log(f'Training/unscaled_dyn_loss', outputs['dyn_loss'] / self.hparams.kl_scaling)
        self.log(f"Training/total_loss", total_loss)
        # for key, value in outputs.items():
        #     print(f"{key}: {value}")
        
        return total_loss
    
    def validation_step(self, batch, batch_idx):
        outputs = self(*batch)
        total_loss = sum(list(val for (key, val) in outputs.items() if not key.endswith('entropy')))
        for key, value in outputs.items():
            self.log(f"Validation/{key}", value)
        self.log(f'Validation/unscaled_dyn_loss', outputs['dyn_loss'] / self.hparams.kl_scaling)
        self.log(f"Validation/total_loss", total_loss)
        
        return total_loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.network.parameters(), lr=self.hparams.learning_rate, weight_decay=self.hparams.weight_decay)
        return optimizer
    

class MLPDecoder(nn.Module):
    def __init__(self, latent_dim, num_vars, output_channels, width=7) -> None:
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
            nn.ConvTranspose2d(16, output_channels, (2,2), (1,1)),
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
    ):
        super().__init__()
        self.num_input_channels = num_input_channels
        self.embedding_dim = embedding_dim
        self.codebook_size = codebook_size
        self.num_variables = num_variables
        self.sparse = sparse
        # self.precomputed_states = dict()
        
        if mlp:
            self.decoders = nn.ModuleList([MLPDecoder(embedding_dim, num_variables, output_channels=1, width=7) for _ in range(num_input_channels)])
        else:
            self.decoders = nn.ModuleList([Decoder(embedding_dim, num_variables, output_channels=1) for _ in range(num_input_channels)])
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
            
            emission_means = self.get_emission_means_sparse(state_bit_vecs, x.device)
            obs_logits = self.compute_obs_logits_sparse(x, emission_means)
        else:
            # assuming a diagonal gaussian with unit variance
            emission_means = self.get_emission_means()
            obs_logits = self.compute_obs_logits(x, emission_means)
        return obs_logits

    def get_emission_means_sparse(
        self, 
        state_bit_vecs: Tensor, 
        device: Optional[Union[str, torch.device]] = None
    ) -> Tensor:
        # TODO add batch support
        states = F.one_hot(state_bit_vecs, num_classes=2).float()
        embeds = torch.einsum("ktc,tce->kte", states, self.latent_embedding)
        emission_probs = torch.stack([decoder(embeds)[:,0] for decoder in self.decoders], dim=1)
        return emission_probs

    def get_emission_means(self):
        z = self.latent_embedding[torch.arange(self.num_variables),self.all_idcs,:]
        emission_probs = torch.stack([decoder(z)[:,0] for decoder in self.decoders], dim=1)
        return torch.reshape(emission_probs, (*(self.codebook_size,)*self.num_variables, *emission_probs.shape[1:]))
        
    def compute_obs_logits_sparse(self, x, emission_means):
        # TODO add batch support
        #TODO separate channels and views rather than treating them interchangably?
        output = - ((emission_means - x[None,0]) ** 2).sum(dim=[-2,-1]) / 2 #- math.log(2*math.pi) / 2 * emission_means.shape[-2] * emission_means.shape[-1]
        return output
    
    def compute_obs_logits(self, x, emission_means):
        #TODO separate channels and views rather than treating them interchangably?
        output = - ((emission_means[None] - x[(slice(None),) + (None,)*self.num_variables]) ** 2).sum(dim=[-2,-1]) / 2 #- math.log(2*math.pi) / 2 * emission_means.shape[-2] * emission_means.shape[-1]
        output = einops.rearrange(output, 'batch ... num_views -> (...) (batch num_views)') # TODO BATCH DIM SHOULD BE SOMEWHERE ELSE
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
            emission_means = torch.stack([self.get_emission_means_sparse(bit_vec, device=latent_dist.device) for bit_vec in bit_vecs], dim=0)
            mean_prediction = (latent_dist[...,None,None,None] * emission_means).sum(dim=1)
        else:
            emission_means = self.get_emission_means()
            emission_means = einops.rearrange(emission_means, '... num_views h w -> (...) num_views h w')
            mean_prediction = (latent_dist[...,None,None,None] * emission_means[None]).sum(dim=1)
                

        return mean_prediction

    @property
    def device(self):
        return next(self.parameters()).device
    


    
