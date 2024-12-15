import torch
from torch.utils.data import Dataset
import math
from typing import List

class EnhancedFeatureDataset(Dataset):
    def __init__(self, tokenizer, target_model, texts: List[str], seq_len=128):
        """
        tokenizer: the tokenizer for the target model
        target_model: a ModelWrapper instance for the target model (with .model and .tokenizer)
        texts: list of raw text strings (one context per line)
        seq_len: max sequence length for truncation/padding
        """
        self.tokenizer = tokenizer
        self.target_model = target_model
        self.seq_len = seq_len
        self.texts = texts

    def __len__(self):
        return len(self.texts)

    @torch.no_grad()
    def __getitem__(self, idx):
        #get raw text then tokenize and pad
        text = self.texts[idx]
        ids = self.tokenizer.encode(text)
        ids = ids[:self.seq_len]
        ids = ids + [self.tokenizer.eos_token_id]*(self.seq_len - len(ids))
        input_ids = torch.tensor(ids, dtype=torch.long).unsqueeze(0).cuda()

        #run target model with hidden states
        #setting output_hidden_states=True returns all layer states
        outputs = self.target_model.model(input_ids, output_hidden_states=True)
        
        hidden_states = outputs.hidden_states #tuple with layers+1 tensors and each has dimension (batch, seq_len, hidden_dim)
        last_hidden = hidden_states[-1]  #dimension (1, seq_len, hidden_dim)

        #average over seq_len to get a single vector for context
        avg_hidden = last_hidden.mean(dim=1)  #dimension (1, hidden_dim)

        #compute qv distribution for last token
        q_v = self.target_model.get_token_distribution(input_ids)  #dimension (1, vocab_size)

        entropy = -torch.sum(q_v * torch.log(q_v + 1e-9))
        entropy = entropy.unsqueeze(0)
        entropy = entropy.unsqueeze(1)

        #combine avg_hidden and entropy into one feature vector
        features = torch.cat([avg_hidden, entropy], dim=-1)  #dimension (1, hidden_dim+1)
        features = features.half()
        
        #returns (input_ids, features)
        #originally the first has dimension (1, seq_len) and I squeeze to (seq_len,)
        #originally the second has dimension (1, hidden_dim+1) and I squeeze to (hidden_dim+1,)
        return input_ids.squeeze(0), features.squeeze(0)

def collate_fn(batch):
    """
    Collate function to stack a list of (input_ids, features) into a batch.
    batch: List of tuples (input_ids, features) from __getitem__

    input_ids: (seq_len,)
    features: (hidden_dim+1,)

    We want:
    input_ids_batch: (batch_size, seq_len)
    features_batch: (batch_size, hidden_dim+1)
    """
    input_ids = torch.stack([b[0] for b in batch], dim=0)
    features = torch.stack([b[1] for b in batch], dim=0)
    return input_ids, features
