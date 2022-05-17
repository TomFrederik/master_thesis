def compute_obs_logits(self, x, emission_means):
    #TODO separate channels and views rather than treating them interchangably?
    output = einops.rearrange(torch.zeros_like(x), '... views h w -> ... views (h w)') - 10 ** 10
    output[torch.arange(len(x)),...,torch.argmin(einops.rearrange(x, '... views h w -> ... (views h w)'), dim=-1)] = 0
    return output