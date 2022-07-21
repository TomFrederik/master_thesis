import itertools
import math

from memory_efficient_attention import efficient_dot_product_attention_pt
import torch
import sys
sys.path.append('../../')
from src.common import Tensor

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
    out[p != 0] = p[p != 0] * (torch.log(p[p != 0]) - torch.log(q[p != 0]))
    out = (out * (p != 0).float()).sum(dim=-1)
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
    posterior: Tensor,
    nonterms: Tensor
) -> Tensor:
    """Computes a * KL(stop_grad(posterior) || prior) + (1-a) * KL(posterior || stop_grad(posterior))

    :param balancing_coeff: Balancing coefficient a.
    :type balancing_coeff: float
    :param prior: Prior belief. Should have shape (*, N).
    :type prior: Tensor
    :param posterior: Posterior belief. Should have shape (*, N).
    :type posterior: Tensor
    :param nonterms: Nonterminals (0 or 1). Should have shape (*,).
    :type nonterms: Tensor
    :return: KL balancing loss.
    :rtype: Tensor
    """
    if balancing_coeff < 0 or balancing_coeff > 1:
        raise ValueError("balancing_coeff must be in [0, 1]")
    
    kl = (balancing_coeff * discrete_kl(posterior.detach(), prior) + (1 - balancing_coeff) * discrete_kl(posterior, prior.detach()))
    return (nonterms * kl).mean()

def _grad_discrete_entropy(p):
    out = torch.zeros_like(p)
    out[p == 0] = 0
    out[p != 0] = p[p != 0].log()
    out[p != 0] = out[p != 0] * p[p != 0]
    return -out.sum(dim=-1).mean(dim=list(range(1, len(p.shape)-1)))

@torch.no_grad()
def _no_grad_discrete_entropy(p): 
    return _grad_discrete_entropy(p)

def discrete_entropy(p, grad=False):
    if grad:
        return _grad_discrete_entropy(p)
    else:
        return _no_grad_discrete_entropy(p)
    

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


def generate_all_combinations(codebook_size, num_variables):
        all_idcs = []
        for i, idcs in enumerate(itertools.product(range(codebook_size), repeat=num_variables)):
            all_idcs.append(list(idcs))
        return torch.tensor(all_idcs, dtype=torch.long)
 
    
def test_generate_all_combinations():
    codebook_size = 3
    num_variables = 2
    
    all_combos = generate_all_combinations(codebook_size, num_variables)
    target = torch.tensor([[0,0],[0,1],[0,2],[1,0],[1,1],[1,2],[2,0],[2,1],[2,2]], dtype=torch.long)
    assert torch.equal(all_combos, target)

def gumbel_softmax(logits, tau=1, hard=False, eps=1e-10, dim=-1):
    # type: (Tensor, float, bool, float, int) -> Tensor
    r"""
    Samples from the `Gumbel-Softmax distribution`_ and optionally discretizes.
    You can use this function to replace "F.gumbel_softmax".
    
    Args:
      logits: `[..., num_features]` unnormalized log probabilities
      tau: non-negative scalar temperature
      hard: if ``True``, the returned samples will be discretized as one-hot vectors,
            but will be differentiated as if it is the soft sample in autograd
      dim (int): A dimension along which softmax will be computed. Default: -1.
    Returns:
      Sampled tensor of same shape as `logits` from the Gumbel-Softmax distribution.
      If ``hard=True``, the returned samples will be one-hot, otherwise they will
      be probability distributions that sum to 1 across `dim`.
    .. note::
      This function is here for legacy reasons, may be removed from nn.Functional in the future.
    .. note::
      The main trick for `hard` is to do  `y_hard - y_soft.detach() + y_soft`
      It achieves two things:
      - makes the output value exactly one-hot
      (since we add then subtract y_soft value)
      - makes the gradient equal to y_soft gradient
      (since we strip all other gradients)
    Examples::
        >>> logits = torch.randn(20, 32)
        >>> # Sample soft categorical using reparametrization trick:
        >>> F.gumbel_softmax(logits, tau=1, hard=False)
        >>> # Sample hard categorical using "Straight-through" trick:
        >>> F.gumbel_softmax(logits, tau=1, hard=True)
    .. _Gumbel-Softmax distribution:
        https://arxiv.org/abs/1611.00712
        https://arxiv.org/abs/1611.01144
    """
    def _gen_gumbels():
        gumbels = -torch.empty_like(logits).exponential_().log()
        if torch.isnan(gumbels).sum() or torch.isinf(gumbels).sum():
            # to avoid zero in exp output
            gumbels = _gen_gumbels()
        return gumbels

    gumbels = _gen_gumbels()  # ~Gumbel(0,1)
    gumbels = (logits + gumbels) / tau  # ~Gumbel(logits,tau)
    y_soft = gumbels.softmax(dim)

    if hard:
        # Straight through.
        index = y_soft.max(dim, keepdim=True)[1]
        y_hard = torch.zeros_like(logits).scatter_(dim, index, 1.0)
        ret = y_hard - y_soft.detach() + y_soft
    else:
        # Reparametrization trick.
        ret = y_soft
    return ret



test_generate_all_combinations()