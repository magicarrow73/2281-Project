import torch
import torch.optim as optim
import torch.nn.functional as F
from models.metrics import compute_distance
from tqdm import tqdm
import logging
from torch.cuda.amp import autocast, GradScaler
import csv
import os
import wandb

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
        drafter_probs.append(q_i)
    q_i_list = torch.stack(drafter_probs, dim=1)
    assert q_i_list.shape[2] == q_v.shape[1]
    return q_v, q_i_list

def train_learner_with_target(learner, drafter_indices, target_model, data_loader, ptfile, metric='kl', epochs=1, lr=1e-5, save_interval=50, model_family="unknown", drafter_indices_str="0", metric_name="kl", timestamp="0000-00-00_00-00-00", sizes=None, L=3, checkpoint_dir='learner_checkpoints'):
    """
    Train the Learner using a pre-generated dataset.
    """
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    optimizer = optim.Adam(learner.parameters(), lr=lr)
    learner.train()

    training_data = torch.load(ptfile)

    scaler = GradScaler()
    epoch_losses = []

    os.makedirs(checkpoint_dir, exist_ok=True)

    final_loss_filename = f"{checkpoint_dir}/{model_family}-{drafter_indices_str}-{metric_name}-{timestamp}-losses.csv"
    with open(final_loss_filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["epoch", "loss"])

    intermediate_loss_filename = f"{checkpoint_dir}/{model_family}-{drafter_indices_str}-{metric_name}-{timestamp}-intermediate-losses.csv"
    with open(intermediate_loss_filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["epoch", "step", "loss"])

    wandb_initialized = wandb.run is not None

    for epoch in range(epochs):
        print(f"\nStarting epoch {epoch+1}/{epochs}...")
        running_loss = 0.0
        count = 0

        interval_loss_sum = 0.0
        interval_count = 0

        data = training_data[epoch]
        for step, d in enumerate(data):
            if step % 500 == 0:
                logging.info(f"Processed {step} batches")
            features = d["features"].to(device)
            if drafter_indices == None:
                d_all = d["d_all"].to(device)
            else:
                d_all = d["d_all"]
                d_all = d_all[:, drafter_indices]

            if sizes == None:
                assert False
            s = torch.tensor(sizes)
            assert s.shape[0] == L
            s = s.reshape(1, -1)
            
            d_all = d_all / s 
            d_all = d_all.to(device)
            optimizer.zero_grad()

            with autocast():
                logits = learner(features)
                L_dist = F.softmax(logits, dim=-1)
                loss = torch.mean(torch.sum(L_dist * d_all, dim=-1))

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            #torch.nn.utils.clip_grad_norm_(learner.parameters(), 1.0)

            running_loss += loss.item()
            count += 1

            interval_loss_sum += loss.item()
            interval_count += 1

            if step > 0 and step % save_interval == 0:
                interval_avg_loss = interval_loss_sum / interval_count
                with open(intermediate_loss_filename, 'a', newline='') as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow([epoch, step, interval_avg_loss])

                if wandb_initialized:
                    wandb.log({"intermediate_loss": interval_avg_loss, 
                               "epoch": epoch+1, 
                               "step": step})

                interval_loss_sum = 0.0
                interval_count = 0

            if step % 1000 == 0 and step > 0:
                avg_current_loss = running_loss / count
                logging.info(f"Epoch {epoch+1}, Step {step}, Current Average Loss: {avg_current_loss:.4f}")
        
        avg_loss = running_loss / count
        epoch_losses.append(avg_loss)
        print(f"Epoch {epoch+1} completed with average loss {avg_loss}")

        with open(final_loss_filename, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([epoch, avg_loss])
        
        if wandb_initialized:
            wandb.log({"epoch_loss": avg_loss, "epoch": epoch+1})

    print(f"Final losses saved to {final_loss_filename}")
    print(f"Intermediate losses saved to {intermediate_loss_filename}")

    return epoch_losses

def sample_training_data(drafters, target_model, data_loader, metric='kl', epochs=1, output="training_data.pt", k=1, sizes = None):
    """
    Generate a pre-computed dataset of features and distances for training a Learner offline so we do not waste resources during inference.

    This function iterates over a dataset multiple times (epochs) and for each batch it does the following:
        Extracts the hidden states of the target model and computes the average last hidden state and the entropy measure.
        Obtains distributions q_v (denoting the target) and q_i (denoting the drafter) for the current input batch.
        Computes distances d_all between drafter model distributions and the target model distribution.
        Scales distances by the relative model sizes if they are given.
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
            s = torch.tensor(sizes)
            assert s.shape[0] == L
            s = s.reshape(1, -1)
            #features = features.half()
            q_v_expanded = q_v.unsqueeze(1).expand_as(q_i_list) 
            d_all = compute_distance(
                q_i_list.reshape(-1, vocab_size),
                q_v_expanded.reshape(-1, vocab_size),
                metric=metric,
                k=k)
            d_all = d_all.reshape(batch_size, L).detach().to(cpu)
            d_all = d_all * s
            d_all = d_all.detach()
            data.append({"features": features, "d_all": d_all})
        training_data.append(data)
    torch.save(training_data, output)
    return training_data

def distill_drafter_with_teacher(student_model_wrapper, teacher_model_wrapper, data_loader, epochs=20, temperature=1.0,
                                lr=1e-5, distillation_directory="distillation_directory", wandb_project=None, wandb_run_name=None):
    """
    Perform distillation by minimizing the KL divergence between distributions.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    student_model = student_model_wrapper.model.to(device)
    teacher_model = teacher_model_wrapper.model.to(device)
    
    teacher_model.eval()
    student_model.train()

    optimizer = optim.Adam(student_model.parameters(), lr=lr)
    #scaler = GradScaler()
    
    if wandb_project:
        if wandb.run is None:
            wandb.init(project=wandb_project, name=wandb_run_name)
        wandb_initialized = True
    else:
        wandb_initialized = False

    #global_step = 0
    for epoch in range(epochs):
        running_loss = 0.0
        count = 0
        for step, input_ids in enumerate(data_loader):
            input_ids = input_ids.to(device)
            optimizer.zero_grad()
            
            #teacher forward pass with next-token prediction
            with torch.no_grad():
                teacher_outputs = teacher_model(input_ids)
                teacher_logits = teacher_outputs.logits[:, -1, :].float()
                teacher_probs = F.softmax((teacher_logits / temperature), dim=-1)
            
            #student forward pass
            student_outputs = student_model(input_ids)
            student_logits = student_outputs.logits[:, -1, :]
            student_log_probs = F.log_softmax((student_logits / temperature), dim=-1)

            #formula from online and kl divergence documentation
            loss = (temperature**2) * F.kl_div(student_log_probs, teacher_probs, reduction='batchmean')
            
            # scaler.scale(loss).backward()
            # scaler.step(optimizer)
            # scaler.update()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student_model.parameters(), max_norm=1.0)
            optimizer.step()

            running_loss += loss.item()
            count += 1
            #global_step += 1

            if step % 200 == 0 and step > 0:
                avg_loss = running_loss / count
                logging.info(f"Epoch {epoch+1}, Step {step}, Current Distillation Loss: {avg_loss:.4f}")

        avg_epoch_loss = running_loss / count
        print(f"Epoch {epoch+1} completed with average distillation loss {avg_epoch_loss:.4f}")
        if wandb_initialized:
            wandb.log({"epoch_distill_loss": avg_epoch_loss, "epoch": epoch+1})

    os.makedirs(distillation_directory, exist_ok=True)
    student_model.save_pretrained(distillation_directory)
    student_model_wrapper.tokenizer.save_pretrained(distillation_directory)

    if wandb_initialized:
        wandb.finish()
    return
