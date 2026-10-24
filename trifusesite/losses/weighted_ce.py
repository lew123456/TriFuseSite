import torch
from torch.nn import functional as F

def weighted_binary_loss(logits, labels, class_weights):
    valid = (labels == 0) | (labels == 1)
    if not valid.any():
        raise ValueError('Protein has no valid binary labels')
    loss_dtype = torch.float32 if logits.dtype in (torch.float16, torch.bfloat16) else logits.dtype
    weights = torch.as_tensor(class_weights, device=logits.device, dtype=loss_dtype)
    if weights.shape != (2,) or not torch.isfinite(weights).all() or (weights <= 0).any():
        raise ValueError('Supply two finite positive class weights: w0, w1')
    y = labels[valid].long()
    losses = F.cross_entropy(logits[valid].to(loss_dtype), y, reduction='none')
    return (losses * weights[y]).sum() / weights[y].sum()
