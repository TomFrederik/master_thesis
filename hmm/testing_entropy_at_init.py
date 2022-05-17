import numpy as np
import torch
import torch.nn as nn

from models import FactorizedTransitionResNet


num_states = 1000
num_features = 128
hidden_dim = 128

in_features = nn.Embedding(num_states, num_features)
out_features = nn.Embedding(num_states, num_features)
nn.init.xavier_uniform_(in_features.weight)
nn.init.xavier_uniform_(out_features.weight)

in_module = FactorizedTransitionResNet(num_features, num_states, hidden_dim)
out_module = FactorizedTransitionResNet(num_features, num_states, hidden_dim)

in_logits = in_module(in_features.weight)
out_logits = out_module(out_features.weight)
print(out_logits.shape)


dist =  torch.softmax(in_logits @ out_logits.T, dim=1)

entropy = torch.sum(-dist * torch.log(dist), dim=1)
print(entropy) # all are very close to 6.9



