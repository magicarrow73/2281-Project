import torch
import torch.optim as optim
import torch.nn.functional as F
from models.metrics import compute_distance
from tqdm import tqdm

@torch.no_grad()
def get_distributions(drafters, target_model, input_ids):
    """
    Compute q_v and q_i for all drafters.

    Returns:
        q_v: (batch, vocab_size) target distribution
        q_i_list: (batch, L, vocab_size) drafter distributions
    """
    q_v = target_model.get_token_distribution(input_ids)
    drafter_probs = []
    for d in drafters:
        q_i = d.get_token_distribution(input_ids)
        drafter_probs.append(q_i) #each q_i has dimension (batch, vocab_size)
    q_i_list = torch.stack(drafter_probs, dim=1) #dimension (batch, L, vocab_size)
    return q_v, q_i_list

def train_learner_with_target(learner, drafters, target_model, data_loader, metric='kl', epochs=1, lr=1e-4):
    """
    Train the Learner to pick a Drafter that best matches the target model's distribution.
    data_loader yields (input_ids, features).

    Steps:
    - For each batch:
      - Compute q_v from target model
      - Compute q_i from each Drafter
      - Compute distance d_all = d(q_i, q_v) for each i
      - Learner outputs L_dist = softmax(logits)
      - Loss = E_{i ~ L_dist}[d(q_i, q_v)] = sum(L_dist * d_all)
    """
    optimizer = optim.Adam(learner.parameters(), lr=lr)
    learner.train()

    epoch_losses = []

    for epoch in range(epochs):

        running_loss = 0.0
        count = 0

        print(f"\nStarting epoch {epoch+1}/{epochs}...")

        for step, (input_ids, features) in enumerate(data_loader):
            if step % 100 == 0:
                logging.info(f"Processed {step} batches")
            input_ids = input_ids.cuda()
            features = features.cuda()
            optimizer.zero_grad()

            #get the distributions
            q_v, q_i_list = get_distributions(drafters, target_model, input_ids)
            #q_v has dimension (batch, vocab_size)
            #q_i_list has dimension (batch, L, vocab_size)

            #distance for each drafter
            #d_all should contain the distances for all of the drafters, and we flatten to run faster
            batch_size, L, vocab_size = q_i_list.size()
            q_v_expanded = q_v.unsqueeze(1).expand_as(q_i_list) #dimension (batch, L, vocab_size)
            d_all = compute_distance(
                q_i_list.reshape(-1, vocab_size),
                q_v_expanded.reshape(-1, vocab_size),
                metric=metric
            )
            d_all = d_all.reshape(batch_size, L) #dimension (batch, L), reshaped from flattened state
            logits = learner(features.half()) #dimension (batch, L)
            L_dist = F.softmax(logits, dim=-1) #dimension (batch, L)

            #take the loss averaged over the batch
            loss = torch.mean(torch.sum(L_dist * d_all, dim=-1))
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            count += 1
        
        avg_loss = running_loss / count
        epoch_losses.append(avg_loss)
        print(f"Epoch {epoch} completed with average loss {avg_loss}")

    return epoch_losses
