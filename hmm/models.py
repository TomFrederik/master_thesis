from collections import deque, namedtuple
import itertools

import einops
import einops.layers.torch as layers
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F

# from .utils import discrete_kl
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
        self.prior = nn.Parameter(torch.ones(self.state_dim, device=device)) / self.state_dim
        
    def forward(self, batch_size):
        return einops.repeat(self.prior, 'dim -> batch_size dim', batch_size=batch_size)

    def to(self, device):
        self.prior = self.prior.to(device)
    
    @property
    def device(self):
        return self.prior.device

# models the transition distribution of the latent space conditioned on the action
# instantiates |A| transition matrices
class ActionConditionedTransition(nn.Module):
    def __init__(self, state_dim, num_actions, device):
        super().__init__()
        self.state_dim = state_dim
        self.num_actions = num_actions
        
        self.matrices = torch.stack([nn.Parameter(torch.ones((self.state_dim, self.state_dim), device=device)) / self.state_dim for _ in range(self.num_actions)], dim=0)
    
    def forward(self, state, action):
        return torch.einsum('bi, bij -> bj', state, self.matrices[action])
    
    def to(self, device):
        self.matrices = self.matrices.to(device)

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
        transition_matrices,
        # value_prefix_predictor,
        autoencoder,
        kl_balancing_coeff,
        l_unroll,
        discount_factor,
        # reward_support,
    ):
        super().__init__()
        self.state_prior = state_prior
        self.transition_matrices = transition_matrices
        # self.value_prefix_predictor = value_prefix_predictor
        self.autoencoder = autoencoder
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

    def forward(self, obs_sequence, action_sequence, reward_sequence):
        outputs = dict()
        
        batch_size, seq_len, *_ = obs_sequence.shape
        # # compute target value prefixes
        # print(f"{reward_sequence.shape = }")
        # target_value_prefixes = self.compute_value_prefixes(reward_sequence)
        # print(f"{target_value_prefixes.shape = }")
        
        # # convert them to classes
        # transformed_target_value_prefixes = self.scalar_transform(target_value_prefixes)
        # target_value_prefixes_phi = self.reward_phi(transformed_target_value_prefixes)
        
        # autoencoder
        obs_logits, latent_dist, latent_loss = self.autoencoder(einops.rearrange(obs_sequence, 'batch seq ... -> (batch seq) ...'))
        obs_logits = einops.rearrange(obs_logits, 'bs ... -> bs (...)') # flatten over latent states
        latent_dist = sum_factored_logits(torch.log(latent_dist))
        recon_loss = obs_logits + latent_dist # TODO: Is this right?
        recon_loss = torch.logsumexp(recon_loss, dim=-1)
        obs_logits = einops.rearrange(obs_logits, '(batch seq) ... -> batch seq ...', batch=batch_size, seq=seq_len)
        latent_dist = einops.rearrange(latent_dist, '(batch seq) ... -> batch seq ...', batch=batch_size, seq=seq_len)
        outputs['recon_loss'] = -recon_loss.mean()
        outputs['latent_loss'] = latent_loss.mean()
        
        # prior
        state_belief = self.state_prior(batch_size)
        posterior_0 = (state_belief.log() + obs_logits[:,0]).exp()
        posterior_0 = posterior_0 / posterior_0.sum(dim=1, keepdim=True).detach()
        prior_loss = discrete_kl(state_belief, posterior_0.log())
        state_belief = posterior_0
        outputs['prior_loss'] = prior_loss
        
        # dynamics
        prior_sequences = deque([[] for i in range(self.l_unroll)], maxlen=self.l_unroll)
        value_prefix_loss = 0
        dyn_loss = 0
        for t in range(1,seq_len):
            # extrapolate l_unroll steps
            state_belief_prior_sequence = self.k_step_extrapolation(state_belief, action_sequence[:, t-1:])
            for i in range(state_belief_prior_sequence.shape[1]):
                prior_sequences[i].append(state_belief_prior_sequence[:,i])
            
            # predict value prefixes
            # value_prefix_logits = self.value_prefix_predictor(torch.cat([state_belief, state_belief_prior_sequence], dim=1))
            # print(f"{value_prefix_logits.shape = }")
            # score them against the actual value prefixes
            # TODO
            # value_prefix_loss += F.cross_entropy(value_prefix_logits, target_value_prefixes_phi[:,t])
            
            # get the priors for the next state
            priors = torch.stack(prior_sequences.popleft(), dim=1)[:,0]
            prior_sequences.append([])
            
            # get the posterior for the next state
            state_belief_posterior = (state_belief.log() + obs_logits[:,t]).exp()
            state_belief_posterior = state_belief_posterior + 1e-10 # TODO: this is very hacky
            state_belief_posterior = state_belief_posterior / state_belief_posterior.sum(dim=1, keepdim=True)
            
            # compute the dynamics loss
            dyn_loss += self.kl_balancing_loss(priors, state_belief_posterior)
            
            # set belief to the posterior
            state_belief = state_belief_posterior
            pass
        
        # take mean of value prefix loss
        outputs['value_prefix_loss'] = value_prefix_loss / seq_len
        
        # take mean of dyn loss
        outputs["dyn_loss"] = dyn_loss / seq_len
        
        return outputs

    def kl_balancing_loss(self, prior, posterior):
        return (self.kl_balancing_coeff * discrete_kl(prior.detach(), posterior.log()) + (1 - self.kl_balancing_coeff) * discrete_kl(prior, posterior.detach().log()))

    def k_step_extrapolation(self, state_belief, action_sequence, k=None):
        if k is None:
            k = self.l_unroll
        state_belief_prior_sequence = []
        for t in range(min(action_sequence.shape[1], k)):
            state_belief = self.transition_matrices(state_belief, action_sequence[:, t])
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
        vae_kwargs,
        kl_balancing_coeff,
        l_unroll,
        discount_factor,
        learning_rate,
        device,
        # reward_support,
    ):
        super().__init__()
        self.save_hyperparameters()
        
        self.prior = StatePrior(state_dim, device)
        self.transition_matrices = ActionConditionedTransition(state_dim, num_actions, device)
        # self.value_prefix_predictor = ValuePrefixPredictor(state_dim, **vp_kwargs)
        self.autoencoder = dVAE(**vae_kwargs) #TODO
        self.network = DiscreteNet(
            self.prior,
            self.transition_matrices,
            # self.value_prefix_predictor,
            self.autoencoder,
            kl_balancing_coeff,
            l_unroll,
            discount_factor,
            # reward_support,
        )
    
    def forward(self, obs, actions, rewards):
        return self.network(obs, actions, rewards)
    
    def training_step(self, batch, batch_idx):
        obs, actions, rewards = batch
        outputs = self(obs, actions, rewards)
        
        for key, value in outputs.items():
            self.log(f"Training/{key}", value)
        
        return sum(list(outputs.values()))
    
    def validation_step(self, batch, batch_idx):
        obs, actions, rewards = batch
        outputs = self(obs, actions, rewards)
        
        for key, value in outputs.items():
            self.log(f"Validation/{key}", value)
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
        one_hot = F.one_hot(idcs, num_classes=self.codebook_size).float().to(self.embeds[0].weight.device)
        z = torch.stack([one_hot[:,i] @ self.embeds[i].weight for i in range(self.num_variables)], dim=1)
        
        return z
        
class Encoder(nn.Module):

    def __init__(self, input_channels=3, num_vars=32, latent_dim=32, codebook_size=32):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv2d(input_channels, 256, kernel_size=7),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            layers.Rearrange('b c h w -> b (c h w)'),
            nn.Linear(256, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, num_vars*codebook_size),
        )

    def forward(self, x):
        out = self.net(x)
        return out

class Decoder(nn.Module):

    def __init__(self, latent_dim=32, num_vars=32, output_channels=3):
        super().__init__()

        self.net = nn.Sequential(
            layers.Rearrange('b n d -> b (n d)'),
            nn.Linear(latent_dim*num_vars, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            layers.Rearrange('b d -> b d 1 1'),
            nn.ConvTranspose2d(256, 16, 7, 2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.ConvTranspose2d(16, output_channels, 1, 1),
        )
        
    def forward(self, x):
        return self.net(x)

    def set_bn_eval(self, m):
        if isinstance(m, nn.modules.batchnorm._BatchNorm):
            m.eval()


    def set_bn_train(self, m):
        if isinstance(m, nn.modules.batchnorm._BatchNorm):
            m.train()
# -----------------------------------------------------------------------------

class dVAE(nn.Module):

    def __init__(
        self, 
        num_input_channels,
        embedding_dim,
        codebook_size,
        num_variables,
        entropy_scale,
    ):
        super().__init__()
        self.num_input_channels = num_input_channels
        self.embedding_dim = embedding_dim
        self.codebook_size = codebook_size
        self.num_variables = num_variables
        
        self.encoder = Encoder(input_channels=num_input_channels, latent_dim=embedding_dim, codebook_size=codebook_size, num_vars=num_variables)
        self.decoder = Decoder(latent_dim=embedding_dim, num_vars=num_variables, output_channels=num_input_channels)
        self.quantizer = TotalQuantizer(num_variables=num_variables, codebook_size=codebook_size, embedding_dim=embedding_dim, entropy_scale=entropy_scale)

    def get_emission_means(self):
        # disable the batchnorms in the decoder
        self.decoder.apply(self.decoder.set_bn_eval)
        # TODO make this more efficient by using batching
        emission_probs = None
        for idcs in itertools.product(range(self.codebook_size), repeat=self.num_variables):
            z = self.quantizer.embed_idcs(torch.Tensor(idcs)[None].long())
            x_dist = self.decoder(z)
            if emission_probs is None:
                emission_probs = torch.empty((*(self.codebook_size,)*self.num_variables, *x_dist.shape[1:]), device=x_dist.device)
            emission_probs[idcs] = x_dist[0]
        
        # enable the batchnorms in the decoder
        self.decoder.apply(self.decoder.set_bn_train)
        
        return emission_probs
        
    def forward(self, x):
        emission_means = self.get_emission_means()
        encoded = self.encoder(x)
        latent_dist, latent_loss = self.quantizer(encoded)
        # latent dist has shape (batch_size, num_variables, codebook_size)
        
        # assuming a diagonal gaussian with unit variance
        obs_logits = self.compute_obs_logits(x, emission_means)
        
        return obs_logits, latent_dist, latent_loss

    def compute_obs_logits(self, x, emission_means):
        return - ((emission_means[None] - x[:,None,None]) ** 2).sum(dim=[-3,-2,-1]) / 2
     
    @torch.no_grad()
    def reconstruct_only(self, x):
        z = self.encoder(x)
        z_q, *_ = self.quantizer(z)
        logits = self.decoder(z_q)
        b, c, h, w = logits.shape
        logits = einops.rearrange(logits, 'b c h w -> (b h w) c')
        x_hat = torch.multinomial(torch.softmax(logits, dim=-1), 1)
        x_hat = einops.rearrange(x_hat, '(b h w) c -> b h (w c)', b=b, h=h, w=w)
        return x_hat
    
    @torch.no_grad()
    def decode_only(self, latent_dist):
        """

        :param latent_dist: shape (batch_size, codebook_size ** num_vars)
        :type latent_dist: torch.Tensor
        :return: x_hat: decoded z
        :rtype: torch.Tensor
        """
        # reshape to (batch_size, codebook_size, codebook_size, codebook_size, ...)
        while len(latent_dist.shape[1:]) < self.num_variables:
            latent_dist = einops.rearrange(latent_dist, 'b ... (codebook_size rest) -> b ... codebook_size rest', codebook_size=self.codebook_size)
        
        # marginalize over latent dist
        latent_dist = self.marginalize(latent_dist)
        
        # sample from latent dist
        sampled_idcs = torch.cat([torch.multinomial(latent_dist[:,i], 1) for i in range(latent_dist.shape[1])], dim=1)
        
        # embed
        embedded_latents = self.quantizer.embed_idcs(sampled_idcs)
        
        # decode
        x_hat = self.decoder(embedded_latents)
        
        return x_hat

    def marginalize(self, latent_dist):
        out = torch.empty((latent_dist.shape[0], len(latent_dist.shape)-1, latent_dist.shape[1]), device=latent_dist.device)
        for dim in range(1, len(latent_dist.shape)):
            # sum over all dimensions excluding dim
            out[:, dim-1] = torch.sum(latent_dist, dim=[d for d in range(1, len(latent_dist.shape)) if d != dim])
        return out
    
    @torch.no_grad()
    def _compute_perplexity(self, ind):
        # debugging: cluster perplexity. when perplexity == num_embeddings then all clusters are used exactly equally
        encodings = F.one_hot(ind, self.quantizer.n_embed).float().reshape(-1, self.quantizer.n_embed)
        avg_probs = encodings.mean(0)
        perplexity = (-(avg_probs * torch.log(avg_probs + 1e-10)).sum()).exp()
        cluster_use = torch.sum(avg_probs > 0)
        return perplexity, cluster_use
    

# value_prefix = ValuePrefixPredictor(state_dim, num_values=601, lstm_hidden_dim=128, num_lstm_layers=1, mlp_hidden_dims=[100,100])

# support = namedtuple('support', ['min', 'max', 'size'])
# reward_support = support(-10, 10, 601)