from sparsemax_k import BitConverter, Tensor, PrioritizedItem, PriorityQueue, get_possible_next_bit_vecs
from time import time
import torch
from lvmhelpers.pbinary_topk import batched_topk

num_bits = 16
k = 10
bitconverter = BitConverter(bits=num_bits, device='cuda')

logits = torch.randn(num_bits, 2, device='cuda')

def binary_topk(logits: Tensor, k: int, bitconverter: BitConverter):
    
    time1 = time()
    #TODO add batch dim support
    if len(logits.shape) < 2 or logits.shape[-1] != 2:
        raise ValueError(f"logits must have shape (..., num_vars, 2), but has shape {logits.shape}")
    if k < 1 or k > 2**logits.shape[0]:
        raise ValueError(f"k must be between 1 and {2**logits.shape[0]}, but is {k}")
    
    
    topk_bit_vecs = []
    
    top_bit_vec = torch.argmax(logits, dim=-1)
    time1 = time()
    top_idx = bitconverter.bitvec_to_idx(top_bit_vec)
    print(f'Time to convert bit vector to idx: {time() - time1}')
    visited= set([top_idx]) # This is where batch support fails.
    next_bit_vecs = PriorityQueue()
    next_bit_vecs.put(PrioritizedItem(0, top_bit_vec, top_idx))
    
    total_time_getting_nexts = 0
    total_time_updating_queue = 0
    total_time_getting_prio = 0
    print(f"time to initialize: {time() - time1}")
    
    while len(topk_bit_vecs) < k:
        time1 = time()
        prio_item = next_bit_vecs.get()
        topk_bit_vecs.append(prio_item.item)
        total_time_getting_prio += time() - time1
        
        time1 = time()
        possibe_nexts = get_possible_next_bit_vecs(logits, prio_item, visited)
        total_time_getting_nexts += time() - time1
        
        time1 = time()
        for item in possibe_nexts:
            idx = item.state_idx
            if idx not in visited:
                visited.add(idx)
                next_bit_vecs.put(item)
        total_time_updating_queue += time() - time1
    print(f"Time to get prio: {total_time_getting_prio}")
    print(f"Time to get possible nexts: {total_time_getting_nexts}")
    print(f"Time to update queue: {total_time_updating_queue}")
    time1 = time()
    out = torch.stack(topk_bit_vecs, dim=0)#, topk_state_idcs
    print(f"Time to stack: {time() - time1}")
    return out

time1 = time()
topk_bit_vecs = binary_topk(logits, k, bitconverter)
print(f"binary_topk: {time() - time1}")
print(topk_bit_vecs)

logits = logits[None, :, 1]
batch_size, latent_size = logits.shape
# get the top-k bit-vectors
time1 = time()
bit_vector_z = torch.empty((batch_size, k, latent_size), dtype=torch.float32)
batched_topk(logits.detach().cpu().numpy(), bit_vector_z.numpy(), k)
bit_vector_z = bit_vector_z.to('cuda').long()
print(f"Time to run batched_topk: {time() - time1}")

print(bit_vector_z)