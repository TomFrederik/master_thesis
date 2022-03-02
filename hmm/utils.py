def discrete_kl(p, logq):
    if len(p.shape) != 2:
        raise ValueError(f"p must be a 2D tensor, but has shape {p.shape}")
    if p.shape != logq.shape:
        raise ValueError("p and logq must have the same shape")
    out = (p * (p.log() - logq))
    out[p==0] = 0
    out = out.sum(dim=[1]).mean()
    return out
