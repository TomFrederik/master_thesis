import einops
import torch
import torch.nn as nn
import torch.nn.functional as F

def norm(batchnorm, num_features):
    if batchnorm:
        return nn.BatchNorm1d(num_features)
    else:
        return nn.Identity()

def construct_mlp(input_dim, mlp_hidden_dims, output_dim, batchnorm):
    mlp_list = [nn.Linear(input_dim, mlp_hidden_dims[0])]
    for i in range(len(mlp_hidden_dims)-1):
        mlp_list.extend([
            nn.ReLU(), 
            norm(batchnorm, mlp_hidden_dims[i]),
            nn.Linear(mlp_hidden_dims[i], mlp_hidden_dims[i+1])
        ])
    mlp_list.extend([
        nn.ReLU(),
        norm(batchnorm, mlp_hidden_dims[-1]),
        nn.Linear(mlp_hidden_dims[-1], output_dim),
    ])
    
    return nn.Sequential(*mlp_list)



class SparseValuePrefixPredictor(nn.Module):
    def __init__(self, num_variables, mlp_hidden_dims, num_values, vp_batchnorm=False):
        super().__init__()
        
        self.mlp_hidden_dims = mlp_hidden_dims
        self.num_values = num_values

        # TODO make embedding dim a parameter
        self.latent_embedding = nn.Parameter(torch.zeros(num_variables, num_values, 128))
        
        # init with std = 1/sqrt(input_dim_to_network) = 1/sqrt(embed_dim * num_variables) 
        nn.init.normal_(self.latent_embedding, mean=0, std=1/(num_variables*128)**0.5)
        
        
        self.mlp = construct_mlp(num_variables*128, self.mlp_hidden_dims, self.num_values, vp_batchnorm)

    def forward(self, bit_vecs):
        
        states = F.one_hot(bit_vecs, num_classes=2).float()
        embeds = torch.einsum("bkdc,dce->bkde", states, self.latent_embedding)
        batch, k, *_ = embeds.shape
        embeds = einops.rearrange(embeds, 'batch k vars dim -> (batch k) (vars dim)')
        out = self.mlp(embeds)
        return einops.rearrange(out, '(batch k) dim -> batch k dim', batch=batch, k=k)[...,0]
    
class OLDSparseValuePrefixPredictor(nn.Module):
    def __init__(self, num_variables, mlp_hidden_dims, num_values, vp_batchnorm=False):
        super().__init__()
        
        self.mlp_hidden_dims = mlp_hidden_dims
        self.num_values = num_values
        
        self.mlp = construct_mlp(num_variables, self.mlp_hidden_dims, self.num_values, vp_batchnorm)

    def forward(self, x):
        return self.mlp(x)[...,0]

class DenseValuePrefixPredictor(nn.Module):
    def __init__(self, state_size, num_values, mlp_hidden_dims, vp_batchnorm=False):
        super().__init__()
        self.state_size = state_size
        self.num_values = num_values
        self.mlp_hidden_dims = mlp_hidden_dims
        
        self.mlp = construct_mlp(state_size, self.mlp_hidden_dims, self.num_values, vp_batchnorm)
            
    def forward(self, x):
        out = self.mlp(x)[...,0]
        return out

    @property
    def device(self):
        return next(self.parameters()).device