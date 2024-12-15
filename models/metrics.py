import torch
import torch.nn.functional as F

def kl_divergence(p, q):
    return torch.sum(p * (torch.log(p + 1e-6) - torch.log(q + 1e-6)), dim=-1)

def l2_distance(p, q):
    return torch.sqrt(torch.sum((p - q)**2, dim=-1))

def chi_squared_distance(p, q):
    return torch.sum((p - q)**2 / (q + 1e-6), dim=-1)

def wasserstein_distance(p, q):
    cdf_p = torch.cumsum(p, dim=-1)
    cdf_q = torch.cumsum(q, dim=-1)
    return torch.sum(torch.abs(cdf_p - cdf_q), dim=-1)

def compute_distance(p, q, metric='kl'):
    if metric == 'kl':
        return kl_divergence(p, q)
    elif metric == 'l2':
        return l2_distance(p, q)
    elif metric == 'chi_squared':
        return chi_squared_distance(p, q)
    elif metric == 'wasserstein':
        return wasserstein_distance(p, q)
    else:
        raise ValueError("Idk we have not defined this distance metric yet")
