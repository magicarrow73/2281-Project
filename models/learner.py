import torch
import torch.nn as nn
import torch.nn.functional as F

class LearnerModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, L):
        super(LearnerModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, L)
    def forward(self, x):
        #x should be of dimension (batch, input_dim)
        h = F.relu(self.fc1(x))
        logits = self.fc2(h) #should be of dimension (batch, L)
        return logits

def sample_drafter(learner_logits):
    probs = F.softmax(learner_logits, dim=-1)
    drafter_idx = torch.multinomial(probs, num_samples=1)
    return drafter_idx.item()
