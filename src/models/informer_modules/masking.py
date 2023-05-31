import torch


def triangular_causal_mask(B, L, device=torch.device("cpu")):
    mask_shape = [B, 1, L, L]
    with torch.no_grad():
        mask = torch.triu(torch.ones(mask_shape, dtype=torch.bool, device=device), diagonal=1)
    return mask


def prob_mask(B, H, L, index, scores, device=torch.device("cpu")):
    mask = torch.ones(L, scores.shape[-1], dtype=torch.bool, device=device).triu(1)
    mask_ex = mask[None, None, :].expand(B, H, L, scores.shape[-1])
    indicator = mask_ex[torch.arange(B)[:, None, None], torch.arange(H)[None, :, None], index, :]
    mask = indicator.view(scores.shape)
    return mask
