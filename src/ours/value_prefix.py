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

    
class MarginalValuePrefixPredictor(nn.Module):
    def __init__(self, num_variables, mlp_hidden_dims, output_dim, vp_batchnorm=False):
        super().__init__()
        
        self.mlp_hidden_dims = mlp_hidden_dims
        self.output_dim = output_dim

        self.mlp = construct_mlp(num_variables, self.mlp_hidden_dims, self.output_dim, vp_batchnorm)

    def forward(self, beliefs, bit_vecs):
        
        marginalized_beliefs = (beliefs[...,None] * bit_vecs).sum(dim=-2)
        # print(f"{beliefs[0] = }")
        out = self.mlp(marginalized_beliefs)
        return out[...,0]