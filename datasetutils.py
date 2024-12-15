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
        input_ids = torch.tensor(ids, dtype=torch.long)
        return input_ids

def collate_fn(batch):
    input_ids = torch.stack(batch, dim=0)
    return input_ids
