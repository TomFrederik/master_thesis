import math

from memory_efficient_attention import efficient_dot_product_attention_pt
import torch

def discrete_kl(p, q):
    if len(p.shape) != 2:
        raise ValueError(f"p must be a 2D tensor, but has shape {p.shape}")
    if p.shape != q.shape:
        raise ValueError("p and q must have the same shape")
    out = torch.zeros_like(p)
    
    out[p == 0] = 0
    out[p != 0] = p[p != 0] * (torch.log(p[p != 0]) - torch.log(q[p != 0]))
    out = out[p != 0].sum() / p.shape[0]
    return out

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
