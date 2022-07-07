from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from commons import Tensor
from sparsemax_k import BitConverter, sparsemax


# NOTE: Non-factorized is equivalent (in terms of no. of free params) to factorized with embedding_dim = |S|
class FactorizedTransition(nn.Module):
    def __init__(
        self, 
        num_variables: int, 
        codebook_size: int, 
        num_actions: int, 
        embedding_dim: Optional[int] = 128, 
        hidden_dim: Optional[int] = 128, 
        layer_dims: Optional[Union[List, Tuple]] = None, 
        sparse: Optional[bool] = False,
        ) -> None:
        
        super().__init__()
        
        self.num_variables = num_variables
        self.codebook_size = codebook_size
        self.num_actions = num_actions
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.layer_dims = layer_dims
        self.sparse = sparse
        
        self.state_emb = nn.Parameter(torch.zeros(codebook_size**num_variables, embedding_dim))
        nn.init.xavier_uniform_(self.state_emb.data) # necessary to get uniform distribution at init
        
        self.key_features = nn.Parameter(torch.zeros(num_actions, embedding_dim, hidden_dim))
        self.query_features = nn.Parameter(torch.zeros(num_actions, embedding_dim, hidden_dim))

        nn.init.xavier_uniform_(self.key_features.data) # necessary to get uniform distribution at init
        nn.init.xavier_uniform_(self.query_features.data) # necessary to get uniform distribution at init

        if sparse:
            self.bitconverter = BitConverter(bits=num_variables, device='cuda' if torch.cuda.is_available() else 'cpu')
        
    def forward(
        self, 
        state_belief: Tensor, 
        action: Tensor, 
        state_bit_vecs: Optional[Tensor] = None,
        ) -> Tuple[Tensor, Union[Tensor, None]]:
        
        if self.sparse:
            return sparse_transition(self.state_emb, self.key_features[action], self.query_features[action], state_belief, state_bit_vecs, self.bitconverter)
        else:
            return dense_transition(self.state_emb, self.key_features[action], self.query_features[action], state_belief)
    
    @property
    def device(self):
        return next(self.parameters()).device

    def build_module(self, hidden_dim, layer_dims):
        net = nn.Sequential()
        if layer_dims is None:
            layer_dims = [hidden_dim, hidden_dim]
        else:
            layer_dims = [hidden_dim] + layer_dims + [hidden_dim]
        
        for i in range(len(layer_dims)-1):
            net.add_module(f"{i}", nn.Linear(layer_dims[i], layer_dims[i+1]))
            if i != len(layer_dims)-2:
                net.add_module(f"{i}_relu", nn.ReLU())
        return net


def dense_transition(
    state_emb: Tensor,
    in_features: Tensor, 
    out_features: Tensor, 
    state_belief: Tensor,
    ) -> Tuple[Tensor, None]:
    
    out_features = torch.einsum('aij,si->asj', out_features, state_emb)
    in_features = torch.einsum('aij,si->asj', in_features, state_emb)
    
    beliefs = F.softmax(torch.einsum('bsi,bti->bst', out_features, in_features), dim=-1)
    beliefs = torch.einsum('ijk,ij->ik', beliefs, state_belief)
    return beliefs, None

def sparse_transition(
    state_emb: Tensor,
    in_features: Tensor, 
    out_features: Tensor, 
    state_belief: Tensor, 
    state_bit_vecs: Tensor, 
    bitconverter: BitConverter, 
    dim: Optional[int] = -1,
    k: Optional[int] = None,
    batching: bool = True,
    batching_size: int = 10000,
    ) -> Tuple[Tensor, Tensor]:
    
    # X_features have shape (num_states, hidden_dim)
    # state_beliefs have shape (1, k)
    # state_bit_vecs have shape (k, num_variables)
    # output should be a tensor of shape (num_states, ) # or maybe (k, ) again after applying sparsemax_k??
    # in_features has shape (num_states, hidden_dim)
    if k is None:
        k = state_bit_vecs.shape[1]
    
    idcs = bitconverter.bitvec_to_idx(state_bit_vecs)

    out_features = torch.einsum('aij,abi->abj', out_features, state_emb[idcs])
    in_features = torch.einsum('aij,si->asj', in_features, state_emb)

    if batching:
        batch, num_states, emb_dim = in_features.shape
        beliefs = []
        for i in range(num_states // batching_size + 1):
            beliefs.append(torch.einsum('abd, acd -> abc', out_features, in_features[:,i*batching_size:(i+1)*batching_size]))
        beliefs = torch.cat(beliefs, dim=-1)
        beliefs = F.softmax(beliefs, dim=-1)
        beliefs = torch.einsum('abc,ab->ac', beliefs, state_belief)
    else:
        beliefs = F.softmax(torch.einsum('abd,acd->abc', out_features, in_features), dim=-1)
        beliefs = torch.einsum('abc,ab->ac', beliefs, state_belief)

    values, indices = torch.topk(beliefs.log(), k=k, dim=dim)

    out = sparsemax(values, dim=dim)

    state_bit_vecs = bitconverter.idx_to_bitvec(indices)

    return out, state_bit_vecs
    
# class ActionConditionedMLPTransition(nn.Module):
#     def __init__(self, state_dim, num_actions, hidden_dim=128):
#         super().__init__()
#         self.state_dim = state_dim
#         self.num_actions = num_actions

#         self.wall_idcs = [(0,3),(1,3),(2,3),(3,3),(4,3),(6,3)]
        
#         # if action == 0: # down
#         #     next_cell[0] += 1
#         # elif action == 1: # up
#         #     next_cell[0] -= 1
#         # elif action == 2: # right
#         #     next_cell[1] += 1
#         # elif action == 3: # left
#         #     next_cell[1] -= 1
#         test = torch.zeros(49).to('cuda')
#         test[43] = 1
#         print(test.reshape((7,7)))

#         down = torch.diag_embed(torch.ones(self.state_dim), offset=-7).to('cuda')
#         down = down[:-7, :-7]
#         down[-7:,-7:] = torch.diag_embed(torch.ones(7)).to('cuda')
#         print((down @ test).reshape((7,7)))

#         right = torch.diag_embed(torch.ones(self.state_dim), offset=-1).to('cuda')
#         right = right[:-1, :-1]
#         right[[6,13,20,27,34,41,48], [6,13,20,27,34,41,48]] = 1
        
#         for (row, col) in self.wall_idcs:
#             idx = col + row * 7
#             right[:,idx-1] = 0
#             right[idx-1,idx-1] = 1
#         print((right @ test).reshape((7,7)))
        
#         left = torch.diag_embed(torch.ones(self.state_dim), offset=1).to('cuda')
#         left = left[:-1, :-1]
#         left[[0,7,14,21,28,35,42], [0,7,14,21,28,35,42]] = 1
        
#         for (row, col) in self.wall_idcs:
#             idx = col + row * 7
#             left[:,idx+1] = 0
#             left[idx+1,idx+1] = 1
#         print((left @ test).reshape((7,7)))
        
#         up = torch.diag_embed(torch.ones(self.state_dim), offset=7).to('cuda')
#         up = up[:-7, :-7]
#         up[:7,:7] = torch.diag_embed(torch.ones(7)).to('cuda')
#         print((up @ test).reshape((7,7)))
#         self.actions = [down, up, right, left]
        
#     def forward(self, state, action):
#         # print(state.reshape((1, 7, 7)))
#         # print(action)
#         out = torch.stack([self.actions[a] @ state[i] for i, a in enumerate(action)], dim=0)
#         out = out / torch.sum(out, dim=-1, keepdim=True)
#         # print(out.reshape((1,7,7)))
#         # raise ValueError
#         return out
