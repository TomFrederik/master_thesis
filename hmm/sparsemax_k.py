from dataclasses import dataclass, field
from queue import PriorityQueue
from typing import Any, TypeVar

from entmax import sparsemax
import pytest
import torch

Tensor = TypeVar('Tensor', bound=torch.Tensor)

@dataclass(order=True)
class PrioritizedItem:
    priority: int
    item: Any=field(compare=False)
    state_idx: int=field(compare=False)

def convert_state_idx_to_bit_vector(state_idx, num_vars, device=None):
    """Converts a state index to a bit vector."""
    if device is None:
        device = 'cpu'
    #TODO make this more efficient
    return torch.tensor([int(n) for n in bin(state_idx)[2:].zfill(num_vars)], device=device)

def binary_topk(logits, k):
    #TODO add batch dim support
    if len(logits.shape) < 2 or logits.shape[-1] != 2:
        raise ValueError(f"logits must have shape (..., num_vars, 2), but has shape {logits.shape}")
    if k < 1 or k > 2**logits.shape[0]:
        raise ValueError(f"k must be between 1 and {2**logits.shape[0]}, but is {k}")
    
    
    topk_bit_vecs = []
    topk_state_idcs = []
    
    top_bit_vec = torch.argmax(logits, dim=-1)
    top_idx = compute_state_idx_from_bit_vector(top_bit_vec)
    visited = set([top_idx]) # This is where batch support fails.
    next_bit_vecs = PriorityQueue()
    next_bit_vecs.put(PrioritizedItem(0, top_bit_vec, top_idx))
    
    while len(topk_bit_vecs) < k:
        prio_item = next_bit_vecs.get()
        topk_bit_vecs.append(prio_item.item)
        topk_state_idcs.append(prio_item.state_idx)
        
        for item in get_possible_next_bit_vecs(logits, prio_item, visited, next_bit_vecs, max_elems=k-len(topk_bit_vecs)):
            idx = item.state_idx
            if idx not in visited:
                visited.add(idx)
                next_bit_vecs.put(item)
    return torch.stack(topk_bit_vecs, dim=0), topk_state_idcs

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

def get_possible_next_bit_vecs(logits, prio_item, visited, prio_queue, max_elems):
    prio, bit_vec, state_idx = prio_item.priority, prio_item.item, prio_item.state_idx
    out = []
    for flip_idx in range(len(bit_vec)):
        candidate_idx = get_flipped_bit_state_idx(bit_vec, state_idx, flip_idx)
        
        if candidate_idx not in visited:
            prio_delta = get_logit_changes(logits, flip_idx, bit_vec)
            candidate = flip_bit(bit_vec, flip_idx)
            out.append(PrioritizedItem(prio + prio_delta, candidate, candidate_idx))
    return out

def compute_state_idx_from_bit_vector(bit_vec):
    return (bit_vec.float() @ 2**(torch.arange(bit_vec.shape[-1], device=bit_vec.device, dtype=torch.float))).int().item()

def sparsemax_k(logits, k, dim=-1):
    """Computes sparsemax_k of the given logits.

    :param logits: Logit tensor of shape (num_vars, 2)
    :type logits: Tensor
    :param k: Cut-off hyperparameter -> max number of active states
    :type k: int
    :return: sparsemax_k(logits) as a tensor of shape (k, ) and a list of state indices corresponding to the k possible active states
    :rtype: Tuple[Tensor, List[int]]
    """
    #TODO add batch support
    topk_bit_vecs, topk_state_idcs = binary_topk(logits, k)
    print(f"{logits = }")
    topk_logits = logits[torch.arange(len(logits)), topk_bit_vecs].sum(dim=-1)
    print('topk_logits', topk_logits)
    out = sparsemax(topk_logits, dim=dim)
    
    return out, topk_state_idcs

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

def test_get_logit_changes():
    pass
    # logits = torch.nn.functional.log_softmax(torch.randn(5,2),dim=1)
    


# how would I write a transition when
# I am given a belief over states with their respective idcs and all other states are implicitly zero
# Now, let's assume that I have to pass the features into some module
def sparse_transition(in_features, out_features, in_module, out_module, state_beliefs, state_idcs, dim=-1):
    # X_features have shape (num_states, hidden_dim)
    # state_beliefs have shape (k, )
    # state_idcs is a list of length k
    # output should be a tensor of shape (num_states, ) # or maybe (k, ) again after applying sparsemax_k??
    # print(f'{out_module = }')
    # print(f'{in_module = }')
    # print('hi')
    # print(state_idcs)
    
    selected_out_features = out_module(out_features[state_idcs])
    # print(f"{selected_out_features.shape = }")
    # NOTE:
    # it seems pretty likely that this is going to blow up
    in_features = in_module(in_features) 
    # print(f"{in_features.shape = }")
    # in_features has shape (num_states, hidden_dim)
    # solutions:
        # batch it
        
    # print(f'{(state_beliefs @ selected_out_features).shape = }')
    
    prior_beliefs = (state_beliefs @ selected_out_features) @ in_features.T
    
    # Hm I think this^ is very wrong. I'm just multiplying probabilities with logits here.
    
    print(f'{prior_beliefs.shape = }')
    # TODO support chunking this or something
    # Okay this is a big problem right now. Posterior beliefs is a dist over the joint
    # i.e. it is NOT factorized into a distribution over variables
    # return sparsemax_k(prior_beliefs, k=len(state_idcs), dim=dim)
    
    # The solution: just apply topk to the joint
    # TODO am I sure that the indices are the same as the indices I compute by hand?
    values, indices = torch.topk(prior_beliefs, k=len(state_idcs), dim=dim)
    print(f"{values = }")
    out = sparsemax(values, dim=dim)
    indices = indices.squeeze()
    # print(out)
    # print(indices)
    return out, indices
    

