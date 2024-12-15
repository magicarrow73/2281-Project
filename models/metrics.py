import torch

def kl_divergence(p, q):
    return torch.sum(p * (torch.log(p + 1e-9) - torch.log(q + 1e-9)), dim=-1)

def l2_distance(p, q):
    return torch.sqrt(torch.sum((p - q)**2, dim=-1))

def compute_distance(p, q, metric='kl'):
    if metric == 'kl':
        return kl_divergence(p, q)
    elif metric == 'l2':
        return l2_distance(p, q)
    else:
        raise ValueError("Idk we have not defined this distance metric yet")
