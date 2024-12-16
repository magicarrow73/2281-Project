import torch
import torch.optim as optim
import torch.nn.functional as F
from models.metrics import compute_distance
from tqdm import tqdm
import logging
from torch.cuda.amp import autocast, GradScaler

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
    assert q_i_list.shape[2] == q_v.shape[1]
    return q_v, q_i_list

def train_learner_with_target(learner, drafter_indices, target_model, data_loader, ptfile, metric='kl', epochs=1, lr=1e-6):
    """
    Train the Learner using a pre-generated dataset.
    """
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    optimizer = optim.Adam(learner.parameters(), lr=lr)
    learner.train()

    scaler = GradScaler()

    epoch_losses = []
    training_data = torch.load(ptfile)

    for epoch in range(epochs):
        running_loss = 0.0
        count = 0

        print(f"\nStarting epoch {epoch+1}/{epochs}...")
        data = training_data[epoch]
        for step, d in enumerate(data):
            if step % 100 == 0:
                logging.info(f"Processed {step} batches")
            features = d["features"].to(device)
            if drafter_indices == None:
                d_all = d["d_all"].to(device)
            else:
                d_all = d["d_all"]
                d_all = d_all[:, drafter_indices]
            d_all = d_all.to(device)
            optimizer.zero_grad()

            #setting output_hidden_states=True returns all layer states
            with autocast():
                # for name, param in learner.named_parameters():
                #     if torch.isnan(param).any():
                #         print(f"NaN in parameter {name}")
                #     if torch.isinf(param).any():
                #         print(f"Inf in parameter {name}")
                logits = learner(features)
                # if step % 100 == 0:
                #     print('d_all is', d_all, 'logits are', logits)
                L_dist = F.softmax(logits, dim=-1)
                loss = torch.mean(torch.sum(L_dist * d_all, dim=-1))

            #loss.backward()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            #torch.nn.utils.clip_grad_norm_(learner.parameters(), 1.0)
            # for name, param in learner.named_parameters():
            #     if torch.isnan(param.grad).any():
            #         print(f"NaN in gradients of {name}")
            #optimizer.step()

            running_loss += loss.item()
            count += 1

            if step % 1000 == 0 and step > 0:
                avg_current_loss = running_loss / count
                logging.info(f"Epoch {epoch+1}, Step {step}, Current Average Loss: {avg_current_loss:.4f}")
        
        avg_loss = running_loss / count
        epoch_losses.append(avg_loss)
        print(f"Epoch {epoch+1} completed with average loss {avg_loss}")

    return epoch_losses

def sample_training_data(drafters, target_model, data_loader, metric='kl', epochs=1, lr=1e-4, output="training_data.pt", k=1, sizes = None):
    """
    Train the Learner to pick a Drafter that best matches the target model's distribution.

    Steps:
    - For each batch:
      - Compute q_v from target model
      - Compute q_i from each Drafter
      - Compute distance d_all = d(q_i, q_v) for each i
      - Learner outputs L_dist = softmax(logits)
      - Loss = E_{i ~ L_dist}[d(q_i, q_v)] = sum(L_dist * d_all)
    """
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    cpu = torch.device("cpu")
    training_data = []
    for epoch in range(epochs):
        print(f"\nStarting epoch {epoch+1}/{epochs}...")

        data = []
        
        for step, input_ids in enumerate(data_loader):
            if step % 100 == 0:
                logging.info(f"Processed {step} batches")

            #run target model with hidden states
            input_ids = input_ids.to(device)

            #setting output_hidden_states=True returns all layer states
            outputs = target_model.model(input_ids, output_hidden_states=True)
            last_hidden = outputs.hidden_states[-1]
            avg_hidden = last_hidden.mean(dim=1)

            #compute qv distribution for last token
            q_v_target = target_model.get_token_distribution(input_ids)
            entropy = -torch.sum(q_v_target * torch.log(q_v_target + 1e-6), dim=-1, keepdim=True)
            features = torch.cat([avg_hidden, entropy], dim=-1).detach().to(cpu)
            q_v, q_i_list = get_distributions(drafters, target_model, input_ids)
            batch_size, L, vocab_size = q_i_list.size()
            if sizes == None:
                assert False
            assert len(sizes) == L
            sizes = torch.tensor(sizes)
            sizes = sizes.reshape(1, -1)
            #features = features.half()
            q_v_expanded = q_v.unsqueeze(1).expand_as(q_i_list) 
            d_all = compute_distance(
                q_i_list.reshape(-1, vocab_size),
                q_v_expanded.reshape(-1, vocab_size),
                metric=metric,
                k=k
            )
            d_all = d_all.reshape(batch_size, L).detach().to(cpu) #dimension (batch, L), reshaped from flattened state
            d_all = d_all * sizes
            d_all = d_all.detach()
            data.append({"features": features, "d_all": d_all})
        training_data.append(data)
    torch.save(training_data, output)
    return training_data

# def serialize_data():
#     for 
