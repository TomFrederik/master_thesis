import torch
import torch.nn as nn

from utils import generate_all_combinations


class ValuePrefixPredictor(nn.Module):
    def __init__(self, num_variables, codebook_size, embedding_dim, num_values, mlp_hidden_dims):
        super().__init__()
        self.num_variables = num_variables
        self.codebook_size = codebook_size
        self.embedding_dim = embedding_dim
        self.num_values = num_values
        self.mlp_hidden_dims = mlp_hidden_dims
        
        self.latent_embedding = nn.ModuleList([nn.Embedding(codebook_size, embedding_dim) for _ in range(num_variables)])
            
        mlp_list = [nn.Linear(embedding_dim*num_variables, self.mlp_hidden_dims[0])]
        for i in range(len(self.mlp_hidden_dims)-1):
            mlp_list.extend([
                nn.ReLU(), 
                nn.Linear(self.mlp_hidden_dims[i], self.mlp_hidden_dims[i+1])
            ])
        mlp_list.extend([
            nn.ReLU(),
            nn.Linear(self.mlp_hidden_dims[-1], self.num_values),
        ])
        self.mlp = nn.Sequential(*mlp_list)
        
        self.all_idcs = generate_all_combinations(codebook_size, num_variables).to(self.device)
        
    def forward(self):
        all_embeds = self.get_all_embeds()
        out = self.mlp(all_embeds)
        return out

    def get_all_embeds(self):
        self.all_idcs = self.all_idcs.to(self.device)
        return torch.cat([self.latent_embedding[j](self.all_idcs[:,j]) for j in range(self.num_variables)], dim=1)

    @property
    def device(self):
        return next(self.parameters()).device

def norm(batchnorm, num_features):
    if batchnorm:
        return nn.BatchNorm1d(num_features)
    else:
        return nn.Identity()


class SparseValuePrefixPredictor(nn.Module):
    def __init__(self, num_variables, mlp_hidden_dims, num_values, vp_batchnorm=False):
        super().__init__()
        
        self.mlp_hidden_dims = mlp_hidden_dims
        self.num_values = num_values
        
        mlp_list = [nn.Linear(num_variables, self.mlp_hidden_dims[0])]
        for i in range(len(self.mlp_hidden_dims)-1):
            mlp_list.extend([
                nn.ReLU(), 
                norm(vp_batchnorm, self.mlp_hidden_dims[i]),
                nn.Linear(self.mlp_hidden_dims[i], self.mlp_hidden_dims[i+1])
            ])
        mlp_list.extend([
            nn.ReLU(),
            norm(vp_batchnorm, self.mlp_hidden_dims[-1]),
            nn.Linear(self.mlp_hidden_dims[-1], self.num_values),
        ])
        self.mlp = nn.Sequential(*mlp_list)

    def forward(self, x):
        return self.mlp(x)[...,0]

class DenseValuePrefixPredictor(nn.Module):
    def __init__(self, state_size, num_values, mlp_hidden_dims, sparse=False, sparsemax_k=None, num_variables=None, vp_batchnorm=False):
        super().__init__()
        self.state_size = state_size
        self.num_values = num_values
        self.mlp_hidden_dims = mlp_hidden_dims
        
        if sparse:
            self.bitvec_embed = nn.Sequential(nn.Linear(num_variables*sparsemax_k, 256), nn.ReLU(), nn.Linear(256, 256), nn.ReLU())
        else:
            self.bitvec_embed = nn.Identity()
            
        # mlp_list = [nn.Linear(2*state_size, self.mlp_hidden_dims[0])]
        mlp_list = [nn.Linear(state_size, self.mlp_hidden_dims[0])]
        for i in range(len(self.mlp_hidden_dims)-1):
            mlp_list.extend([
                nn.ReLU(), 
                norm(vp_batchnorm, self.mlp_hidden_dims[i]),
                nn.Linear(self.mlp_hidden_dims[i], self.mlp_hidden_dims[i+1])
            ])
        mlp_list.extend([
            nn.ReLU(),
            norm(vp_batchnorm, self.mlp_hidden_dims[-1]),
            nn.Linear(self.mlp_hidden_dims[-1], self.num_values),
        ])
        self.mlp = nn.Sequential(*mlp_list)
        
    def forward(self, x):
        out = self.mlp(x)[...,0]
        return out

    @property
    def device(self):
        return next(self.parameters()).device