from typing import Optional

import einops
import einops.layers as layers
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import Tensor, generate_all_combinations


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
        )
        
        
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
    
