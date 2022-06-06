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
        # for m in self.latent_embedding:
        #     nn.init.xavier_uniform_(m.weight)
            
        mlp_list = [nn.Linear(embedding_dim*num_variables, self.mlp_hidden_dims[0])]
        for i in range(len(self.mlp_hidden_dims)-1):
            mlp_list.extend([
                # nn.BatchNorm1d(self.mlp_hidden_dims[i]), 
                nn.ReLU(), 
                nn.Linear(self.mlp_hidden_dims[i], self.mlp_hidden_dims[i+1])
            ])
        mlp_list.extend([
            # nn.BatchNorm1d(self.mlp_hidden_dims[-1]),
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
