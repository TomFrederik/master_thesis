from dataclasses import dataclass, field
from queue import PriorityQueue
from typing import Any, TypeVar, Union

import einops
import pytest
import torch
import torch.nn.functional as F
from entmax import sparsemax
from lvmhelpers.pbinary_topk import batched_topk

from commons import Tensor

class BitConverter:
    def __init__(self, bits, device=None):
        self.bits = bits
        self.mask = 2**torch.arange(self.bits).long().to(device)

    def bitvec_to_idx(self, bit_vec: Tensor) -> Tensor:
        return (bit_vec.float() @ self.mask.float()).long()   

    def idx_to_bitvec(self, x):
        # from https://stackoverflow.com/questions/55918468/convert-integer-to-pytorch-tensor-of-binary-bits
        return x.unsqueeze(-1).bitwise_and(self.mask).ne(0).long()


def binary_topk(logits: Tensor, k: int, bitconverter: BitConverter):
    #TODO add batch dim support
    if len(logits.shape) < 2 or logits.shape[-1] != 2:
        raise ValueError(f"logits must have shape (..., num_vars, 2), but has shape {logits.shape}")
    if k < 1 or k > 2**logits.shape[0]:
        raise ValueError(f"k must be between 1 and {2**logits.shape[0]}, but is {k}")
    
    
    logits = logits[None, :, 1]
    batch_size, latent_size = logits.shape
    
    bit_vector_z = torch.empty((batch_size, k, latent_size), dtype=torch.float32)
    batched_topk(logits.detach().cpu().numpy(), bit_vector_z.numpy(), k)
    bit_vector_z = bit_vector_z.to('cuda').long()
    return bit_vector_z

def sparsemax_k(logits, k, bitconverter: BitConverter, dim=-1):
    """Computes sparsemax_k of the given logits.
    """
    #TODO add batch support
    topk_bit_vecs = binary_topk(logits, k, bitconverter)[0]
    topk_logits = logits[torch.arange(len(logits)), topk_bit_vecs].sum(dim=-1)
    out = sparsemax(topk_logits, dim=dim)
    
    return out, topk_bit_vecs
