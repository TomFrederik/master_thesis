from typing import Optional, Union

import einops
import torch
import torch.nn as nn

from src.common import Tensor


class StatePrior(nn.Module):
    def __init__(
        self, 
        num_variables: int, 
        codebook_size: int, 
        device: Optional[Union[torch.device, str]] = None,
        force_uniform: bool = False,
        noise_scale: float = 0.0,
    ) -> None:
        super().__init__()
        self.num_variables = num_variables
        self.codebook_size = codebook_size
        self.state_dim = codebook_size ** codebook_size
        self.force_uniform = force_uniform
        self.noise_scale = noise_scale

        # init prior to uniform distribution
        self.prior = nn.Parameter(torch.zeros((num_variables, codebook_size), device=device))
        
    def forward(self, batch_size: int) -> Tensor:
        out = einops.repeat(self.prior, 'num_vars codebook -> batch_size num_vars codebook', batch_size=batch_size)
        if self.force_uniform:
            out = out.detach()
            out = out + torch.randn_like(out) * self.noise_scale
        return out

    def to(self, device: Union[torch.device, str]) -> None:
        self.prior = self.prior.to(device)
    
    @property
    def device(self) -> torch.device:
        return self.prior.device
