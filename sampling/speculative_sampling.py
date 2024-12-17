import torch
from tqdm import tqdm
import torch
from typing import List
from sampling.kvcache_model import KVCacheModel
from sampling.utils import norm_logits, sample, max_fn
import torch.nn.functional as F
from globals import Decoder
from torch.cuda.amp import autocast
import numpy as np

@torch.no_grad()
def speculative_sampling_v3(prefix : torch.Tensor, approx_models: List[torch.nn.Module], target_model : torch.nn.Module, learner : torch.nn.Module, 
                         max_len : int , gamma : int = 4,
                         temperature : float = 1, top_k : int = 0, top_p : float = 0, random_seed : int = None) -> torch.Tensor:
    """
    DeepMind version Speculative Sampling.
    Accelerating Large Language Model Decoding with Speculative Sampling
    https://arxiv.org/abs/2302.01318
    No KV Cache Optimization
    
    Args:
        x (torch.Tensor): input sequence, (batch, prefix_seqlen), Note that the batch dim is always 1 now.
        approx_model (torch.nn.Module): approx model, the small one
        target_model (torch.nn.Module): target model, the large one
        max_len (int): the max overall generated tokens number.
        gamma (int): $\gamma$, the token number small model guesses.
        temperature (float, optional): Defaults to 1.
        top_k (int, optional): Defaults to 0.
        top_p (float, optional): Defaults to 0.

    Returns:
        torch.Tensor: generated tokens (batch, target_seqlen)
    """
    temperature = 1
    top_p = 0
    top_k = 0
    loops = 0
    # seq_len parameters
    seq_len = prefix.shape[1]
    T = seq_len + max_len
    
    # set learner model parameters
    # input_dim = 4097 
    # hidden_dim = 128
    # L = 3 # 
    # num_layers = 3
    # dropout = 0.2
    assert prefix.shape[0] == 1, "input batch size must be 1"

    # with tqdm(total=T, desc="speculative sampling") as pbar:
    past_key_values = None
    outputs = target_model.model(prefix, output_hidden_states=True)
    last_hidden = outputs.hidden_states[-1]
    avg_hidden = last_hidden.mean(dim=1)
    past_key_values = outputs.past_key_values

    #compute qv distribution for last token
    q_v_target = target_model.get_token_distribution(prefix)
    q_v_target = q_v_target.clamp_min(5e-8)

    entropy = -torch.sum(q_v_target * torch.log(q_v_target + 1e-6), dim=-1, keepdim=True)
    features = torch.cat([avg_hidden, entropy], dim=-1)
    new_sample = False
    p = outputs.logits
    new_start = 0
    while prefix.shape[1] < T:
        loops += 1
        ### q = M_q[prefix + x_0, x_1, .., x_(gamma-2)]
        ### get features of target model
        with autocast():
            # get target model output + states
            
                
            # sample drafter via learner model
            learner_logits = learner(features)
            idx = learner.sample_drafter(learner_logits)
            # idx = 2
            chosen_model = approx_models[idx]
            
            
            x = prefix
            prefix_len = prefix.shape[1]
            for _ in range(gamma):
                # p.logits shape (batch, seq, vocab)
                
                q = chosen_model.model(x).logits
                next_tok = sample(norm_logits(q[:, -1, :], 
                                temperature, top_k, top_p))
                x = torch.cat((x, next_tok), dim=1)
            # normalize the logits
            for i in range(q.shape[1]):
                q[:,i,:] = norm_logits(q[:,i,:],
                                temperature, top_k, top_p)
            
            
            # p  = M_p[prefix + x_0, x_0, .., x_(gamma-1)]
            if not new_sample:
                t_output = target_model.model(x[:, prefix_len:], past_key_values = past_key_values, output_hidden_states = True, use_cache=True)
                p = torch.cat((p, t_output.logits), dim = 1)
                last_hidden = torch.cat((last_hidden, t_output.hidden_states[-1]), dim=1)
            else:
                t_output = target_model.model(x[:, prefix_len - 1:], past_key_values = past_key_values, output_hidden_states = True, use_cache=True)
                p = torch.cat((p, t_output.logits), dim = 1)
                last_hidden = torch.cat((last_hidden, t_output.hidden_states[-1]), dim=1)
                new_start -= 1
            past_key_values = t_output.past_key_values
            for i in range(new_start, p.shape[1]):
                p[:,i,:] = norm_logits(p[:,i,:],
                                temperature, top_k, top_p)
            # n the end position of the valid prefix
            # x = x_[:prefix_len-1] + x_0, ... x_(gamma-1)
            
            is_all_accept = True
            n = prefix_len - 1
            for i in range(gamma):
                if random_seed:
                    torch.manual_seed(random_seed)
                r = torch.rand(1, device = p.device)
                j = x[:, prefix_len + i]
                
                if r < torch.min(torch.tensor([1], device=q.device), p[:, n, j] / q[:, n, j]):
                    # accept, and update n
                    n += 1
                else:
                    # reject
                    t = sample(max_fn(p[:, n, :] - q[:, n, :]))
                    is_all_accept = False
                    break
            prefix = x[:, :n + 1]
            past_key_values = tuple(map(lambda p: (p[0][:,:,:n+1], p[1][:,:n+1,:]), past_key_values))
            p = p[:,:n+1,:]
            # for i in range(len(past_key_values)):
            #     past_key_values[i][0] = past_key_values[i][0][:, :, :n + 1]
            #     past_key_values[i][1] = past_key_values[i][1][:, :, :n + 1]
            last_hidden = last_hidden[:,:n+1,:]
            hidden = last_hidden[:, max(0, n+1-gamma):n+1,:]
            avg_hidden = hidden.mean(dim=1)

            #compute qv distribution for last token
            q_v_target = p[:,n,:]
            q_v_target = q_v_target.clamp_min(5e-8)

            entropy = -torch.sum(q_v_target * torch.log(q_v_target + 1e-6), dim=-1, keepdim=True)
            features = torch.cat([avg_hidden, entropy], dim=-1)
            if is_all_accept:
                t = sample(p[:, -1, :])
            new_sample = True
            
            prefix = torch.cat((prefix, t), dim=1)
            new_start = prefix.shape[1]
        # pbar.update(n - pbar.n)
    return loops, prefix

@torch.no_grad()
def speculative_sampling_v4(prefix : torch.Tensor, approx_models: List[torch.nn.Module], target_model : torch.nn.Module, learner : torch.nn.Module, 
                         max_len : int , gamma : int = 4,
                         temperature : float = 1, top_k : int = 0, top_p : float = 0, random_seed : int = None) -> torch.Tensor:
    """
    DeepMind version Speculative Sampling.
    Accelerating Large Language Model Decoding with Speculative Sampling
    https://arxiv.org/abs/2302.01318
    No KV Cache Optimization
    
    Args:
        x (torch.Tensor): input sequence, (batch, prefix_seqlen), Note that the batch dim is always 1 now.
        approx_model (torch.nn.Module): approx model, the small one
        target_model (torch.nn.Module): target model, the large one
        max_len (int): the max overall generated tokens number.
        gamma (int): $\gamma$, the token number small model guesses.
        temperature (float, optional): Defaults to 1.
        top_k (int, optional): Defaults to 0.
        top_p (float, optional): Defaults to 0.

    Returns:
        torch.Tensor: generated tokens (batch, target_seqlen)
    """
    temperature = 1
    top_p = 0
    top_k = 0
    rejections = 0
    # seq_len parameters
    seq_len = prefix.shape[1]
    T = seq_len + max_len
    
    # set learner model parameters
    # input_dim = 4097 
    # hidden_dim = 128
    # L = 3 # 
    # num_layers = 3
    # dropout = 0.2
    assert prefix.shape[0] == 1, "input batch size must be 1"

    # with tqdm(total=T, desc="speculative sampling") as pbar:
    outputs = target_model.model(prefix, output_hidden_states=True)
    last_hidden = outputs.hidden_states[-1]
    avg_hidden = last_hidden.mean(dim=1)

    #compute qv distribution for last token
    q_v_target = target_model.get_token_distribution(prefix)
    q_v_target = q_v_target.clamp_min(5e-8)

    entropy = -torch.sum(q_v_target * torch.log(q_v_target + 1e-6), dim=-1, keepdim=True)
    features = torch.cat([avg_hidden, entropy], dim=-1)
    p = outputs.logits
    while prefix.shape[1] < T:
        ### q = M_q[prefix + x_0, x_1, .., x_(gamma-2)]
        ### get features of target model
        with autocast():
            # get target model output + states
            
                
            # sample drafter via learner model
            learner_logits = learner(features)
            idx = learner.sample_drafter(learner_logits)
            print(idx)
            # idx = 3
            # print(idx)
            chosen_model = approx_models[idx]
            
            
            x = prefix
            prefix_len = prefix.shape[1]
            for _ in range(gamma):
                # p.logits shape (batch, seq, vocab)
                
                q = chosen_model.model(x).logits
                next_tok = sample(norm_logits(q[:, -1, :], 
                                temperature, top_k, top_p))
                x = torch.cat((x, next_tok), dim=1)
            # normalize the logits
            for i in range(q.shape[1]):
                q[:,i,:] = norm_logits(q[:,i,:],
                                temperature, top_k, top_p)
            
            
            # p  = M_p[prefix + x_0, x_0, .., x_(gamma-1)]

            t_output = target_model.model(x, output_hidden_states = True)
            p = t_output.logits
            for i in range(p.shape[1]):
                p[:,i,:] = norm_logits(p[:,i,:],
                                temperature, top_k, top_p)
            # n the end position of the valid prefix
            # x = x_[:prefix_len-1] + x_0, ... x_(gamma-1)
            
            is_all_accept = True
            n = prefix_len - 1
            for i in range(gamma):
                if random_seed:
                    torch.manual_seed(random_seed)
                r = torch.rand(1, device = p.device)
                j = x[:, prefix_len + i]
                
                if r < torch.min(torch.tensor([1], device=q.device), p[:, n, j] / q[:, n, j]):
                    # accept, and update n
                    n += 1
                else:
                    # reject
                    t = sample(max_fn(p[:, n, :] - q[:, n, :]))
                    is_all_accept = False
                    rejections += 1
                    break
            prefix = x[:, :n + 1]
            # for i in range(len(past_key_values)):
            #     past_key_values[i][0] = past_key_values[i][0][:, :, :n + 1]
            #     past_key_values[i][1] = past_key_values[i][1][:, :, :n + 1]
            last_hidden = t_output.hidden_states[-1][:,:n+1,:]
            avg_hidden = last_hidden.mean(dim=1)

            #compute qv distribution for last token
            q_v_target = p[:,n,:]
            q_v_target = q_v_target.clamp_min(5e-8)

            entropy = -torch.sum(q_v_target * torch.log(q_v_target + 1e-6), dim=-1, keepdim=True)
            features = torch.cat([avg_hidden, entropy], dim=-1)
            if is_all_accept:
                t = sample(p[:, -1, :])
            
            prefix = torch.cat((prefix, t), dim=1)
        # pbar.update(n - pbar.n)
    print(rejections)
    return prefix


@torch.no_grad()
def speculative_sampling(prefix : torch.Tensor, approx_model : torch.nn.Module, target_model : torch.nn.Module, 
                         max_len : int , gamma : int = 4,
                         temperature : float = 1, top_k : int = 0, top_p : float = 0, verbose : bool = False, random_seed : int = None) -> torch.Tensor:
    """
    Google version Speculative Sampling.
    https://arxiv.org/pdf/2211.17192.pdf
        
    Adapted with KV Cache Optimization.
        
    Args:
        x (torch.Tensor): input sequence, (batch, prefix_seqlen), Note that the batch dim is always 1 now.
        approx_model (torch.nn.Module): approx model, the small one
        target_model (torch.nn.Module): target model, the large one
        max_len (int): the max overall generated tokens number.
        gamma (int): $\gamma$, the token number small model guesses.
        temperature (float, optional): Defaults to 1.
        top_k (int, optional): Defaults to 0.
        top_p (float, optional): Defaults to 0.

    Returns:
        torch.Tensor: generated tokens (batch, target_seqlen)
    """
    seq_len = prefix.shape[1]
    T = seq_len + max_len
    
    assert prefix.shape[0] == 1, "input batch size must be 1"

    assert approx_model.device == target_model.device
    
    device = target_model.device
    
    approx_model_cache = KVCacheModel(approx_model, temperature, top_k, top_p)
    target_model_cache = KVCacheModel(target_model, temperature, top_k, top_p)
    
    resample_count = 0
    target_sample_count = 0
    accepted_count = 0
    
    while prefix.shape[1] < T:
        # q = M_q[prefix + x_0, x_1, .., x_(gamma-2)]
        prefix_len = prefix.shape[1]

        x = approx_model_cache.generate(prefix, gamma)
        _ = target_model_cache.generate(x, 1)
        
        n = prefix_len + gamma - 1
        

        for i in range(gamma):
            if random_seed:
                torch.manual_seed(random_seed)
            r = torch.rand(1, device = device)
            j = x[:, prefix_len + i]
            
            if r > (target_model_cache._prob_history[:, prefix_len + i - 1, j]) / (approx_model_cache._prob_history[:, prefix_len + i - 1, j]):
                # reject
                n = prefix_len + i - 1
                break
            
            if verbose:
                print(f"approx guess accepted {j[0]}: \033[31m{Decoder().decode(torch.tensor([j]))}\033[0m")

            accepted_count += 1
        
        # print(f"n : {n}, i : {i}, prefix_len + gamma - 1: {prefix_len + gamma - 1}")
        assert n >= prefix_len - 1, f"n {n}, prefix_len {prefix_len}"
        prefix = x[:, :n + 1]
        
        approx_model_cache.rollback(n+1)
        
        assert approx_model_cache._prob_history.shape[-2] <= n + 1, f"approx_model prob list shape {approx_model_cache._prob_history.shape}, n {n}"
        
        if n < prefix_len + gamma - 1:
            # reject someone, sample from the pos n
            t = sample(max_fn(target_model_cache._prob_history[:, n, :] - approx_model_cache._prob_history[:, n, :]))
            if verbose:
                print(f"target resamples at position {n}: \033[34m{Decoder().decode(t)}\033[0m")
            resample_count += 1
            target_model_cache.rollback(n+1)
        else:
            # all approx model decoding accepted
            assert n == target_model_cache._prob_history.shape[1] - 1
            t = sample(target_model_cache._prob_history[:, -1, :])
            if verbose:
                print(f"target samples {n}: \033[35m{Decoder().decode(t)}\033[0m")
            target_sample_count += 1
            target_model_cache.rollback(n+2)
        
        
        prefix = torch.cat((prefix, t), dim=1)

    if verbose:
        print(f"generated tokens numbers {prefix.shape[-1] - seq_len}, accepted_count {accepted_count}, target_sample_count {target_sample_count}, resample_count {resample_count}")
    return prefix


@torch.no_grad()
def speculative_sampling_v2(prefix : torch.Tensor, approx_model : torch.nn.Module, target_model : torch.nn.Module, 
                         max_len : int , gamma : int = 4,
                         temperature : float = 1, top_k : int = 0, top_p : float = 0, random_seed : int = None) -> torch.Tensor:
    """
    DeepMind version Speculative Sampling.
    Accelerating Large Language Model Decoding with Speculative Sampling
    https://arxiv.org/abs/2302.01318
    No KV Cache Optimization
    
    Args:
        x (torch.Tensor): input sequence, (batch, prefix_seqlen), Note that the batch dim is always 1 now.
        approx_model (torch.nn.Module): approx model, the small one
        target_model (torch.nn.Module): target model, the large one
        max_len (int): the max overall generated tokens number.
        gamma (int): $\gamma$, the token number small model guesses.
        temperature (float, optional): Defaults to 1.
        top_k (int, optional): Defaults to 0.
        top_p (float, optional): Defaults to 0.

    Returns:
        torch.Tensor: generated tokens (batch, target_seqlen)
    """
    seq_len = prefix.shape[1]
    T = seq_len + max_len
    
    assert prefix.shape[0] == 1, "input batch size must be 1"

    with tqdm(total=T, desc="speculative sampling") as pbar:
        while prefix.shape[1] < T:
            # q = M_q[prefix + x_0, x_1, .., x_(gamma-2)]
            x = prefix
            prefix_len = prefix.shape[1]
            for _ in range(gamma):
                # p.logits shape (batch, seq, vocab)
                q = approx_model(x).logits
                next_tok = sample(norm_logits(q[:, -1, :], 
                                  temperature, top_k, top_p))
                x = torch.cat((x, next_tok), dim=1)
            
            # normalize the logits
            for i in range(q.shape[1]):
                q[:,i,:] = norm_logits(q[:,i,:],
                                temperature, top_k, top_p)
            # p  = M_p[prefix + x_0, x_0, .., x_(gamma-1)]
            p = target_model(x).logits
            for i in range(p.shape[1]):
                p[:,i,:] = norm_logits(p[:,i,:],
                                temperature, top_k, top_p)

            # n the end position of the valid prefix
            # x = x_[:prefix_len-1] + x_0, ... x_(gamma-1)
            
            is_all_accept = True
            n = prefix_len - 1
            for i in range(gamma):
                if random_seed:
                    torch.manual_seed(random_seed)
                r = torch.rand(1, device = p.device)
                j = x[:, prefix_len + i]
                
                if r < torch.min(torch.tensor([1], device=q.device), p[:, prefix_len + i - 1, j] / q[:, prefix_len + i - 1, j]):
                    # accept, and update n
                    n += 1
                else:
                    # reject
                    t = sample(max_fn(p[:, n, :] - q[:, n, :]))
                    is_all_accept = False
                    break
         
            prefix = x[:, :n + 1]
            
            if is_all_accept:
                t = sample(p[:, -1, :])
            
            prefix = torch.cat((prefix, t), dim=1)
            pbar.update(n - pbar.n)

    return prefix

