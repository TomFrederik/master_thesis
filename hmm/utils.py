import math

from memory_efficient_attention import efficient_dot_product_attention_pt
import torch
from commons import Tensor

def discrete_kl(p: Tensor, q: Tensor) -> Tensor:
    """Compute the KL divergence between two discrete probability distributions.
    Interprets last dimension of p and q as probabilities.
    Takes the mean over all preceding dimensions.

    :param p: Tensor of shape (..., num_values)
    :type p: Tensor
    :param q: Tensors of shape (..., num_values)
    :type q: Tensor
    :raises ValueError: Divergent shapes
    :raises ValueError: _description_
    :return: KL(p || q)
    :rtype: Tensor
    """
    if p.shape != q.shape:
        raise ValueError("p and q must have the same shape")
    
    out = torch.zeros_like(p)
    out[p == 0] = 0
    out[p != 0] = p[p != 0] * (torch.log(p[p != 0]) - torch.log(q[p != 0]))
    out = out[p != 0].sum(dim=-1).mean()
    return out

# def test_discrete_kl():
#     p = torch.tensor([[0.1, 0.2, 0.3, 0.4], [0.1, 0.2, 0.3, 0.4]])
#     q = torch.tensor([[0.1, 0.2, 0.3, 0.4], [0.1, 0.2, 0.3, 0.4]])
#     assert all(torch.isclose(discrete_kl(p, q), torch.zeros(1)))
    
#     p = torch.tensor([[[1., 0.]]])
#     q = torch.tensor([[[1., 0.]]])
#     assert all(torch.isclose(discrete_kl(p, q), torch.zeros(1)))
    
def kl_balancing_loss(
    balancing_coeff: float, 
    prior: Tensor, 
    posterior: Tensor
) -> Tensor:
    """Computes a * KL(stop_grad(posterior) || prior) + (1-a) * KL(posterior || stop_grad(posterior))

    :param balancing_coeff: Balancing coefficient a.
    :type balancing_coeff: float
    :param prior: Prior belief. Should have shape (*, N).
    :type prior: Tensor
    :param posterior: Posterior belief. Should have shape (*, N).
    :type posterior: Tensor
    :return: KL balancing loss.
    :rtype: Tensor
    """
    if balancing_coeff < 0 or balancing_coeff > 1:
        raise ValueError("balancing_coeff must be in [0, 1]")
    
    return (balancing_coeff * discrete_kl(posterior.detach(), prior) + (1 - balancing_coeff) * discrete_kl(posterior, prior.detach()))

@torch.no_grad()
def discrete_entropy(p):
    out = torch.zeros_like(p)
    out[p == 0] = 0
    out[p != 0] = p[p != 0].log()
    return -(out[p != 0] * p[p != 0]).sum(dim=-1).mean()

def batched_query_attention(keys, queries, query_batch_size=-1):
    if query_batch_size == -1:
        query_batch_size = queries.shape[0]
    out = []
    for i in range(math.ceil(queries.shape[0] / query_batch_size)):
        start = i * query_batch_size
        stop = (i + 1) * query_batch_size
        out.append(attention(keys, queries[start:stop]))
    return torch.cat(out, dim=0)

def attention(keys, queries):
    # print(keys.shape, queries.shape)
    return torch.softmax(torch.einsum('ij,kj->ik', queries, keys) / (queries.shape[1] ** 0.5), dim=-1)

def mem_efficient_attention(keys, queries, query_chunk_size=1024, key_chunk_size=4096):
    return efficient_dot_product_attention_pt(queries, keys, query_chunk_size=query_chunk_size, key_chunk_size=key_chunk_size)
