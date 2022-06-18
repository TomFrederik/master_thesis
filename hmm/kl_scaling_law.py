import matplotlib.pyplot as plt
import torch
from tqdm import tqdm

from utils import discrete_kl


def sample_dist(ndims):
    return torch.randn(ndims)

def kl_dist(ndims):
    dist1 = sample_dist(ndims)
    dist2 = sample_dist(ndims)
    dist1 = torch.softmax(dist1, dim=0)
    dist2 = torch.softmax(dist2, dim=0)
    return discrete_kl(dist1, dist2)

def estimate_kl(ndims, nsamples):
    return torch.mean(torch.stack([kl_dist(ndims) for _ in range(nsamples)]))

kl_estimates = []
ndims_range = list(range(10, 1000, 10))
for ndims in tqdm(ndims_range):
    kl_estimates.append(estimate_kl(ndims, 1000).item())
    

plt.scatter(ndims_range, kl_estimates)
plt.xlabel('Number of dimensions')
plt.ylabel('KL')
plt.show()
