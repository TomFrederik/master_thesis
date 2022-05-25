import einops
import torch
import torch.nn as nn
import torch.nn.functional as F

from commons import Tensor
from sparsemax_k import sparse_transition, BitConverter


class MatrixTransition(nn.Module):
    def __init__(self, num_states, num_actions, device, factorized=False):
        super().__init__()
        self.num_states = num_states
        self.num_actions = num_actions
        self.factorized = factorized
        
        if factorized:
            raise NotImplementedError
        else:
            self.matrices = nn.Parameter(torch.zeros((self.num_actions, self.num_states, self.num_states), device=device))
    
    def forward(self, state, action, *args):
        if self.factorized:
            raise NotImplementedError
        else:
            matrix = self.matrices[action]
            return torch.einsum('bi, bij -> bj', state, torch.softmax(matrix, dim=-1)), None
    
    def to(self, device):
        self.matrices = self.matrices.to(device)

class MLPMatrixTransition(nn.Module):
    def __init__(self, num_states, num_actions, hidden_dim=128):
        super().__init__()
        self.num_states = num_states
        self.num_actions = num_actions
        
        self.action_emb = nn.Embedding(num_actions, hidden_dim)
        for m in self.action_emb:
            nn.init.xavier_uniform_(m.weight)
        
        # self.state_emb = nn.Linear(num_states, hidden_dim)
        
        self.mlp = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, num_states**2))
        
        # self.mlp = nn.Sequential(
        #     nn.Linear(hidden_dim*2, hidden_dim*2),
        #     nn.ReLU(),
        #     nn.Linear(hidden_dim*2, num_states)
        # )

    def forward(self, state, action, *args):
        action = self.action_emb(action)
        # state = self.state_emb(state)
        # return torch.softmax(self.mlp(torch.cat([state, action], dim=-1)), dim=-1)
        transition_matrix = einops.rearrange(self.mlp(action), 'batch (inp out) -> batch inp out', inp=state.shape[-1], out=state.shape[-1])
        return torch.einsum('bi, bij -> bj', state, torch.softmax(transition_matrix, dim=-1)), None

class FactorizedTransition(nn.Module):
    def __init__(self, num_variables, codebook_size, num_actions, embedding_dim=128, hidden_dim=128, layer_dims=None, batch_size=-1, sparse=False):
        super().__init__()
        self.num_variables = num_variables
        self.codebook_size = codebook_size
        self.num_actions = num_actions
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.layer_dims = None
        self.sparse = sparse
        
        self.state_emb = nn.Parameter(torch.zeros(codebook_size**num_variables, embedding_dim))
        # self.state_emb2 = nn.Embedding(num_states, embedding_dim)
        nn.init.xavier_uniform_(self.state_emb.data) # necessary to get uniform distribution at init
        # nn.init.xavier_uniform_(self.state_emb2.weight) # necessary to get uniform distribution at init
        # self.action_emb = nn.Embedding(num_actions, embedding_dim)
        
        self.keys = nn.ModuleList([StateFeatures(embedding_dim, hidden_dim, layer_dims) for _ in range(num_actions)])
        self.queries = nn.ModuleList([StateFeatures(embedding_dim, hidden_dim, layer_dims) for _ in range(num_actions)])
        
        if sparse:
            self.bitconverter = BitConverter(bits=num_variables, device='cuda' if torch.cuda.is_available() else 'cpu')
        
    def forward(self, state_belief, action, state_idcs=None):
        # action = self.action_emb(action)
        # transition_matrix = self._get_transition_matrix(action)
        if self.sparse:
            return sparse_transition(self.state_emb, self.state_emb, self.keys[action], self.queries[action], state_belief, state_idcs, self.bitconverter)
        else:
            return self.dense_transition(self.state_emb, self.state_emb, self.keys[action], self.queries[action], state_belief)
    
    @staticmethod
    def dense_transition(in_features, out_features, in_module, out_module, state_belief):
        out_features = out_module(out_features)
        in_features = in_module(in_features)
        
        prior_beliefs = state_belief @ F.softmax((out_features @ in_features.T), dim=-1)
        
        return prior_beliefs, None

class StateFeatures(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, layer_dims=None):
        super().__init__()
        self.net = nn.Sequential()
        if layer_dims is None:
            layer_dims = [embedding_dim, hidden_dim]
        else:
            layer_dims = [embedding_dim] + layer_dims + [hidden_dim]
        
        for i in range(len(layer_dims)-1):
            self.net.add_module(f"{i}", nn.Linear(layer_dims[i], layer_dims[i+1]))
            if i != len(layer_dims)-2:
                self.net.add_module(f"{i}_relu", nn.ReLU())
        
    def forward(self, state):
        return self.net(state)

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
