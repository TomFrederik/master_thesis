from dataclasses import dataclass, field
from queue import PriorityQueue
from typing import Any, TypeVar

from entmax import sparsemax
import einops
import pytest
import torch
import torch.nn.functional as F

from commons import Tensor

@dataclass(order=True)
class PrioritizedItem:
    priority: int
    item: Any=field(compare=False)
    state_idx: int=field(compare=False)

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
    
    
    topk_bit_vecs = []
    
    top_bit_vec = torch.argmax(logits, dim=-1)
    top_idx = bitconverter.bitvec_to_idx(top_bit_vec)
    visited = set([top_idx]) # This is where batch support fails.
    next_bit_vecs = PriorityQueue()
    next_bit_vecs.put(PrioritizedItem(0, top_bit_vec, top_idx))
    
    while len(topk_bit_vecs) < k:
        prio_item = next_bit_vecs.get()
        topk_bit_vecs.append(prio_item.item)
        # topk_state_idcs.append(prio_item.state_idx)
        
        for item in get_possible_next_bit_vecs(logits, prio_item, visited):
            idx = item.state_idx
            if idx not in visited:
                visited.add(idx)
                next_bit_vecs.put(item)
    return torch.stack(topk_bit_vecs, dim=0)#, topk_state_idcs

def flip_bit(bit_vec: Tensor, flip_idx: int) -> Tensor:
    """Flips the flip_idx-th bit of the given bit vector.

    :param bit_vec: 1D bit vector
    :type bit_vec: Tensor
    :param flip_idx: index at which to flip the bit
    :type flip_idx: int
    :raises ValueError: Bit vector has wrong shape
    :raises ValueError: Unacceptable index
    :return: Vector that is the same as bit_vec, except the flip_idx-th bit is flipped
    :rtype: Tensor
    """
    if len(bit_vec.shape) != 1:
        raise ValueError(f"bit_vec must be a 1D tensor, but has shape {bit_vec.shape}")
    if flip_idx < 0 or flip_idx >= len(bit_vec):
        raise ValueError(f"flip_idx must be between 0 and {len(bit_vec)-1}, but is {flip_idx}")
    cloned = bit_vec.clone()
    cloned[flip_idx] = 1 - cloned[flip_idx]
    return cloned

def get_logit_changes(logits: Tensor, idx: int, bit_vec: Tensor) -> float:
    """Computes difference in state score when flipping the bit of bit_vec at idx. 

    :param logits: Logit tensor of shape (num_vars, 2)
    :type logits: Tensor
    :param idx: Index at which to flip the bit
    :type idx: int
    :param bit_vec: Bit vector to apply the flip to. Shape (num_vars, )
    :type bit_vec: Tensor
    :return: Change in logits resulting from flipping the bit
    :rtype: float
    """
    if len(logits.shape) != 2:
        raise ValueError(f"logits must be a 2D tensor, but has shape {logits.shape}")
    if idx < 0 or idx >= logits.shape[0]:
        raise ValueError(f"idx must be between 0 and {logits.shape[0]-1}, but is {idx}")
    return logits[idx, bit_vec[idx]] - logits[idx, 1-bit_vec[idx]]

def get_flipped_bit_state_idx(old_bit_vec, old_state_idx, flip_idx):
    if old_bit_vec[flip_idx] == 0:
        idx = old_state_idx + 2 ** flip_idx
    else:
        idx = old_state_idx - 2 ** flip_idx
    return idx

def get_possible_next_bit_vecs(logits, prio_item, visited):
    prio, bit_vec, state_idx = prio_item.priority, prio_item.item, prio_item.state_idx
    out = []
    for flip_idx in range(len(bit_vec)):
        candidate_idx = get_flipped_bit_state_idx(bit_vec, state_idx, flip_idx)
        
        if candidate_idx not in visited:
            prio_delta = get_logit_changes(logits, flip_idx, bit_vec)
            candidate = flip_bit(bit_vec, flip_idx)
            out.append(PrioritizedItem(prio + prio_delta, candidate, candidate_idx))
    return out

def sparsemax_k(logits, k, bitconverter: BitConverter, dim=-1):
    """Computes sparsemax_k of the given logits.

    :param logits: Logit tensor of shape (num_vars, 2)
    :type logits: Tensor
    :param k: Cut-off hyperparameter -> max number of active states
    :type k: int
    :return: sparsemax_k(logits) as a tensor of shape (k, ) and a list of state indices corresponding to the k possible active states
    :rtype: Tuple[Tensor, List[int]]
    """
    #TODO add batch support
    topk_bit_vecs = binary_topk(logits, k, bitconverter)
    # print(f"{logits = }")
    topk_logits = logits[torch.arange(len(logits)), topk_bit_vecs].sum(dim=-1)
    # print(f"{topk_logits = }")
    out = sparsemax(topk_logits, dim=dim)
    
    return out, topk_bit_vecs

###
# Tests
###

# def test_binary_topk():
#     logits = torch.tensor([[0.1, 0.9], [0.2, 0.8],[0.3,0.7]]).log()
    
#     topk_idcs, topk_values = binary_topk(logits, 1)
#     assert topk_idcs[0].tolist() == [1,1,1]
#     assert torch.isclose(topk_values[0],torch.sum(logits[:,1]))
    
#     topk_idcs, topk_values = binary_topk(logits, 2)
#     sols = torch.stack([torch.tensor([1,1,1]),torch.tensor([1,1,0])], dim=0)
#     assert torch.isclose(topk_idcs, sols).all()
#     assert torch.isclose(topk_values[0],torch.sum(logits[torch.arange(3),[1,1,1]]))
#     assert torch.isclose(topk_values[1],torch.sum(logits[torch.arange(3),[1,1,0]]))
    
def test_flip_bit():
    for i in range(10):
        test_tensor = torch.zeros(10, dtype=torch.long)
        correct = test_tensor.clone()
        correct[i] = 1
        assert torch.isclose(flip_bit(test_tensor, i), correct).all()
        
        test_tensor = torch.ones(10, dtype=torch.long)
        correct = test_tensor.clone()
        correct[i] = 0
        assert torch.isclose(flip_bit(test_tensor, i), correct).all()
        
    with pytest.raises(ValueError):
        flip_bit(test_tensor, -1)
        flip_bit(test_tensor, 10)
        flip_bit(torch.stack([test_tensor,test_tensor]), 0)

def test_binary_top_k():
    bitconverter = BitConverter(bits=3)
    
    logits = torch.tensor([[0.1, 0.9], [0.2, 0.8],[0.3,0.7]]).log()
    topk_bit_vecs = binary_topk(logits, 1, bitconverter)
    
    topk_bit_vecs = binary_topk(logits, 2, bitconverter)
    sols = torch.stack([torch.tensor([1,1,1]),torch.tensor([1,1,0])], dim=0)
    assert torch.isclose(topk_bit_vecs, sols).all()

    
test_binary_top_k()

