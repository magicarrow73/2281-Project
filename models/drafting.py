from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import torch.nn.functional as F

class ModelWrapper:
    def __init__(self, model_name):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            #load_in_8bit=True,
            device_map="auto",
            trust_remote_code=True
        )
    def get_token_distribution(self, input_ids):
        with torch.no_grad():
            outputs = self.model(input_ids)
            last_token_logits = outputs.logits[:, -1, :]
            probs = F.softmax(last_token_logits, dim=-1)
        return probs
    def forward(self, input_ids, past_key_values = None, use_cache = False):
        return self.model(input_ids, past_key_values = past_key_values, use_cache=use_cache)
