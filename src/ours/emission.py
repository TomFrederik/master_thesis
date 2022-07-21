from typing import Optional, Tuple

import einops
import einops.layers.torch as layers
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from hmm.utils import Tensor, generate_all_combinations


### stolen from dreamer
def conv_out(h_in, padding, kernel_size, stride):
    # can also use this for pooling
    return int((h_in + 2. * padding - (kernel_size - 1.) - 1.) / stride + 1.)

def convtrans_out(h_in, padding, kernel_size, stride):
    return int(((h_in - 1) * stride - 2 * padding + (kernel_size - 1.)) + 1.)

def conv_out_shape(h_in, padding, kernel_size, stride):
    return tuple(conv_out(x, padding, kernel_size, stride) for x in h_in)

def convtrans_out_shape(h_in, padding, kernel_size, stride):
    return tuple(convtrans_out(x, padding, kernel_size, stride) for x in h_in)

class Interpolate(nn.Module):
    def __init__(self, size=None, scale_factor=None, mode='nearest-exact', align_corners=None):
        super().__init__()
        if size is None and scale_factor is None:
            raise ValueError("One of size or scale_factor must be defined")
        self.size = size
        self.scale_factor = scale_factor
        self.mode = mode
        self.align_corners = align_corners
    
    def forward(self, x):
        return F.interpolate(x, self.size, self.scale_factor, self.mode, self.align_corners)

class Decoder(nn.Module):

    def __init__(
        self, 
        latent_dim: int = 32, 
        num_vars: int = 32, 
        output_channels: int = 3, 
        depth: int = 16, 
        img_shape: Optional[Tuple[int, int]] = None,
        scale: int = 1, 
        kernel_size: int = 3,
        latent_dim_is_total: bool = False,
    ) -> None:
        super().__init__()
        
        # dreamer
        if img_shape is None:
            img_shape = (7, 7)
        h, w = img_shape
        
        output_shape = (1, h*scale, w*scale)

        d = depth
        k  = kernel_size
        ctr = 0
        
        shapes = [(2, 2)]
        while shapes[-1] < output_shape[1:]:
            shapes.append((shapes[-1][0]*2, shapes[-1][1]*2))
            ctr += 1
        shapes[-1] = output_shape[1:]
        
        if not latent_dim_is_total:
           latent_dim = num_vars * latent_dim  
        
        # self.linear = nn.Linear(latent_dim, np.prod(self.conv_shape).item())
        self.linear = nn.Linear(latent_dim, 2*2*2**ctr*d)

        transposed_list =[]
        for i in range(ctr):
            transposed_list.append(Interpolate(size=shapes[i+1]))
            transposed_list.append(nn.Conv2d(2**(ctr-i)*d, 2**(ctr-i-1)*d, kernel_size=k, stride=1, padding=1))
            # transposed_list.append(nn.ConvTranspose2d(2**(ctr-i)*d, 2**(ctr-i-1)*d, kernel_size=k, stride=1, padding=1))
            transposed_list.append(nn.ReLU())
        

        self.net = nn.Sequential(
            self.linear,
            layers.Rearrange('b (d h w) -> b d h w', d=2**ctr*d, h=2, w=2),
            *transposed_list,
            # nn.ConvTranspose2d(d, output_channels, k, 2, output_padding=output_paddings[-1]),
            nn.Conv2d(d, output_channels, 1, 1),
        )
        
        
    def forward(self, x):
        if len(x.shape) == 3:
            x = einops.rearrange(x, 'b n d -> b (n d)')
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
        num_views: int,
        embedding_dim: int,
        codebook_size: int,
        num_variables: int,
        sparse: Optional[bool] = False,
        img_shape: Optional[Tuple[int, int]] = None,
        scale: int = 1,
        kernel_size: int = 3,
        depth: int = 16,
    ):
        super().__init__()
        self.num_views = num_views
        self.embedding_dim = embedding_dim
        self.codebook_size = codebook_size
        self.num_variables = num_variables
        self.sparse = sparse
        self.kernel_size = kernel_size
        
        self.decoders = nn.ModuleList([Decoder(
            embedding_dim, 
            num_variables, 
            output_channels=1, 
            img_shape=img_shape, 
            scale=scale, 
            kernel_size=kernel_size, 
            depth=depth
        ) for _ in range(num_views)])
        self.latent_embedding = nn.Parameter(torch.zeros(num_variables, codebook_size, embedding_dim))
        
        # init with std = 1/sqrt(input_dim_to_network) = 1/sqrt(embed_dim * num_variables) 
        nn.init.normal_(self.latent_embedding, mean=0, std=1/(num_variables*embedding_dim)**0.5)
        
        print(self)
        if not sparse:
            self.all_idcs = generate_all_combinations(codebook_size, num_variables)
        

    def forward(
        self, 
        x, 
        state_bit_vecs: Optional[Tensor] = None,
        view_masks: Tensor = None,
    ) -> Tensor:
        if self.sparse:
            if state_bit_vecs is None:
                raise ValueError('state_bit_vecs must be provided for sparse model')
            emission_means = self.get_emission_means_sparse(state_bit_vecs[:,None])
            obs_logits = self.compute_obs_logits_sparse(x, emission_means[:,0], view_masks)
        else:
            # assuming a diagonal gaussian with unit variance
            emission_means = self.get_emission_means()
            obs_logits = self.compute_obs_logits(x, emission_means, view_masks)
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
        emission_probs = torch.cat([decoder(z) for decoder in self.decoders], dim=1)
        return emission_probs
        
    def compute_obs_logits_sparse(self, x, emission_means, view_masks):
        #TODO separate channels and views rather than treating them interchangably?
        view_masks = torch.stack([torch.from_numpy(view_masks[i]).to(self.device) for i in range(len(view_masks))], dim=0)
        output = - ((emission_means - x[:,None]) ** 2 * view_masks[None, None]).sum(dim=[-2,-1]) / 2
        return output
    
    def compute_obs_logits(self, x, emission_means, view_masks):
            
        #TODO separate channels and views rather than treating them interchangably?
        view_masks = torch.stack([torch.from_numpy(view_masks[i]).to(self.device) for i in range(len(view_masks))], dim=0)

        # print(f"{view_masks = }")
        # print(f"{x = }")
        # print(f"{emission_means = }")
        # raise ValueError
        # print(x[:,None].shape)
        # print(emission_means[None].shape)
        # print(view_masks[None,None].shape)
        output = - ((emission_means[None] - x[:,None]) ** 2 * view_masks[None, None]).sum(dim=[-2,-1]) / 2
        # print(output.shape)
        # raise ValueError
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
    
