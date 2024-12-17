import torch
import torch.nn as nn
import torch.nn.functional as F

class LearnerModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, L, num_layers, dropout):
        super(LearnerModel, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(input_dim, hidden_dim))
        for _ in range(num_layers - 1):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))

        self.output_layer = nn.Linear(hidden_dim, L)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
            x = F.relu(x)
            x = self.dropout(x)
        logits = self.output_layer(x)
        return logits

    def sample_drafter(self, learner_logits):
        probs = F.softmax(learner_logits, dim=-1)
        drafter_idx = torch.multinomial(probs, num_samples=1)
        return drafter_idx.item()
