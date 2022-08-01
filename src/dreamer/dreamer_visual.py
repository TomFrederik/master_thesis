import numpy as np
import torch
import torch.nn as nn
from einops.layers.torch import Rearrange


class ObsEncoder(nn.Module):
    def __init__(self, input_shape):
        """
        :param input_shape: tuple containing shape of input
        :param embedding_size: Supposed length of encoded vector
        """
        super(ObsEncoder, self).__init__()

        self.w = input_shape[-1]

        if self.w == 64:
            self.encoder = nn.Sequential(
                nn.Conv2d(1, 32, 4, 2),
                nn.ReLU(),
                nn.Conv2d(32, 64, 4, 2),
                nn.ReLU(),
                nn.Conv2d(64, 128, 4, 2),
                nn.ReLU(),
                nn.Conv2d(128, 256, 4, 2),
                nn.ReLU(),
                Rearrange('b c h w -> b (c h w)'),
            )
        elif self.w == 7:
            self.encoder = nn.Sequential(
                nn.Conv2d(1, 32, 3, 2),
                nn.ReLU(),
                nn.Conv2d(32, 64, 3, 2),
                nn.ReLU(),
                Rearrange('b c h w -> b (c h w)'),
            )
        else:
            raise NotImplementedError


    def forward(self, obs):
        return self.encoder(obs)

    @property
    def embed_size(self):
        if self.w == 64:
            return 1024
        elif self.w == 7:
            return 64
        else:
            raise NotImplementedError

class ObsDecoder(nn.Module):
    def __init__(self, output_shape, input_dim):
        """
        :param output_shape: tuple containing shape of output obs
        :param embed_size: the size of input vector, for dreamerv2 : modelstate 
        """
        super(ObsDecoder, self).__init__()
        c, h, w = output_shape
        
        if h == 64:
            self.decoder = nn.Sequential(
                nn.Linear(input_dim, 1024),
                Rearrange("b d -> b d 1 1"),
                nn.ConvTranspose2d(1024, 128, 5, 2),
                nn.ReLU(),
                nn.ConvTranspose2d(128, 64, 5, 2),
                nn.ReLU(),
                nn.ConvTranspose2d(64, 32, 6, 2),
                nn.ReLU(),
                nn.ConvTranspose2d(32, 1, 6, 2),
            )
        elif h == 7:
            self.decoder = nn.Sequential(
                nn.Linear(input_dim, 64),
                Rearrange("b d -> b d 1 1"),
                nn.ConvTranspose2d(64, 32, 3, 2),
                nn.ReLU(),
                nn.ConvTranspose2d(32, 1, 3, 2),
            )   
        else:
            raise NotImplementedError

    def forward(self, x):
        return self.decoder(x)
    
def conv_out(h_in, padding, kernel_size, stride):
    return int((h_in + 2. * padding - (kernel_size - 1.) - 1.) / stride + 1.)

def output_padding(h_in, conv_out, padding, kernel_size, stride):
    return h_in - (conv_out - 1) * stride + 2 * padding - (kernel_size - 1) - 1

def conv_out_shape(h_in, padding, kernel_size, stride):
    return tuple(conv_out(x, padding, kernel_size, stride) for x in h_in)

def output_padding_shape(h_in, conv_out, padding, kernel_size, stride):
    return tuple(output_padding(h_in[i], conv_out[i], padding, kernel_size, stride) for i in range(len(h_in)))
