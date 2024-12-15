import torch
import argparse
import contexttimer
from colorama import Fore, Style
from transformers import AutoTokenizer, AutoModelForCausalLM
import os
import logging
import sys

from sampling import autoregressive_sampling, speculative_sampling, speculative_sampling_v2
from globals import Decoder
#from accelerate import Accelerator

from models.learner import LearnerModel, sample_drafter
from models.drafting import ModelWrapper
from models.training import train_learner_with_target, get_distributions, sample_training_data
from torch.utils.data import Dataset, DataLoader
from datasetutils import EnhancedFeatureDataset, collate_fn
from datetime import datetime
from datasets import load_dataset
import csv
#import torch.multiprocessing as mp

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

# my local models
MODELZOO = {
    # llama-1
    # https://huggingface.co/PY007/TinyLlama-1.1B-step-50K-105b
    "llama1b": "/share_nfs/fangjiarui/root/code/hf_models/TinyLlama-1.1B-step-50K-105b",
    "llama7b": "/share_nfs/tianzhi/code/llama-7b",
    "llama30b": "/share_nfs/fangjiarui/root/code/hf_models/llama-30b-hf",
    "llama2-7b" : "/share_nfs/fangjiarui/root/code/hf_models/llama-2-7b-hf",
    "llama2-70b" : "/share_nfs/fangjiarui/root/code/hf_models/llama-2-70b-hf",
    "bloom-560m": "/share_nfs/fangjiarui/root/code/hf_models/bloom-560m",
    "bloom-7b": "/share_nfs/fangjiarui/root/code/hf_models/bloomz-7b1",
    "baichuan-7b": "/share_nfs/duanqiyuan/models/source_models/hf/baichuan-7B",
    "baichuan-13b": "/share_nfs/duanqiyuan/models/source_models/hf/Baichuan-13B-Base",
}

def parse_arguments():
    parser = argparse.ArgumentParser(description='args for main.py')

    parser.add_argument('--input', type=str, default="Any recommendations for my holidays in Abu Dhabi?")
    parser.add_argument('--approx_model_name', type=str, default='bigscience/bloom-560m')
    parser.add_argument('--target_model_name', type=str, default='bigscience/bloomz-7b1')
    parser.add_argument('--verbose', '-v', action='store_true', default=False, help='enable verbose mode')
    parser.add_argument('--seed', '-s', type=int, default=None, help='set a random seed, which can makes the result reproducible')
    parser.add_argument('--benchmark', '-b', action='store_true', default=False, help='show benchmark results.')
    parser.add_argument('--profiling', '-p', action='store_true', default=False, help='collect torch profiler results.')
    parser.add_argument('--max_tokens', '-M', type=int, default=20, help='max token number generated.')
    parser.add_argument('--gamma', '-g', type=int, default=4, help='guess time.')

    parser.add_argument('--mode', type=str, default='decode', choices=['decode', 'train_learner', 'create_dataset'], help='Choose mode: decode, train_learner, or create_dataset')
    parser.add_argument('--drafters', nargs='*', help='List of drafters', required=False)
    parser.add_argument('--drafters_idx', nargs="*", help='List of drafter indices for training', required = False)
    parser.add_argument('--ptfile', type=str, help='ptfile', required=False)
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs for learner training')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size for learner training')
    parser.add_argument('--metric', type=str, default='kl', choices=['kl','l2'], help='Distance metric for learner')

    args = parser.parse_args()
    return args

def color_print(text):
    print(Fore.RED + text + Style.RESET_ALL)
    
def benchmark(fn, print_prefix, use_profiler=True, *args, **kwargs):
    TEST_TIME = 10
    profile_filename = f"./profile_logs/{print_prefix}"
    
    with contexttimer.Timer() as t:
        if use_profiler:
            with torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CUDA],
                schedule=torch.profiler.schedule(wait=0, warmup=1, active=2, repeat=1, skip_first=0),
                on_trace_ready=torch.profiler.tensorboard_trace_handler(profile_filename),
                record_shapes=False,
                profile_memory=False,
                # with_stack=True
            ) as prof:
                for _ in range(TEST_TIME): 
                    output = fn(*args, **kwargs)
                    prof.step()
        else:
            for _ in range(TEST_TIME): 
                output = fn(*args, **kwargs)

    print(f"\n [benchmark] {print_prefix}, tokens/sec: {len(output[0]) / t.elapsed / TEST_TIME}, {t.elapsed / TEST_TIME} sec generates {len(output[0])} tokens")

def generate(input_text, approx_model_name, target_model_name, num_tokens=20, gamma = 4,
             random_seed = None, verbose = False, use_benchmark = False, use_profiling = False):
    # NOTE() approx_model_name and target_model_name should use the same tokenizer!
    
    torch_device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    tokenizer = AutoTokenizer.from_pretrained(approx_model_name, trust_remote_code=True)
  
    Decoder().set_tokenizer(tokenizer)
    
    print(f"begin loading models: \n {approx_model_name} \n {target_model_name}")
    small_model = AutoModelForCausalLM.from_pretrained(approx_model_name, 
                                                       torch_dtype=torch.float16,
                                                       #device_map="auto",
                                                       device_map="cuda",
                                                       #load_in_8bit=True,
                                                       #offload_folder="offload",
                                                       trust_remote_code=True)
    large_model = AutoModelForCausalLM.from_pretrained(target_model_name, 
                                                       torch_dtype=torch.float16,
                                                       #device_map="auto",
                                                       device_map="cuda",
                                                       #load_in_8bit=True,
                                                       #offload_folder="offload",
                                                       trust_remote_code=True)
    print("finish loading models")
    
    input_ids = tokenizer.encode(input_text, return_tensors='pt').to(torch_device)

    top_k = 20
    top_p = 0.9

    torch.manual_seed(123)
    output = autoregressive_sampling(input_ids, large_model, num_tokens, top_k = top_k, top_p=top_p)
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    color_print(f"large (target) model autoregressive_sampling: {generated_text}")
    
    if use_benchmark:
        benchmark(autoregressive_sampling, "AS_large", use_profiling,
                  input_ids, large_model, num_tokens, top_k = top_k, top_p=top_p)

    torch.manual_seed(123)
    output = autoregressive_sampling(input_ids, small_model, num_tokens, top_k = top_k, top_p=top_p)
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    color_print(f"small (approx) model autoregressive_sampling: {generated_text}")
    
    if use_benchmark:
        benchmark(autoregressive_sampling, "AS_small", use_profiling,
                  input_ids, small_model, num_tokens, top_k = top_k, top_p=top_p)
    
    torch.manual_seed(123)
    output = speculative_sampling_v2(input_ids, small_model, large_model, num_tokens, top_k = top_k, top_p=top_p, random_seed = random_seed)
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    color_print(f"deepmind's speculative_sampling: {generated_text}")   

    torch.manual_seed(123)
    output = speculative_sampling(input_ids, small_model, large_model, num_tokens, gamma = gamma, top_k = top_k, top_p=top_p, random_seed = random_seed, verbose = verbose)
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    color_print(f"google's speculative_sampling: {generated_text}")
    
    if use_benchmark:
        benchmark(speculative_sampling, "SP", use_profiling,
                  input_ids, small_model, large_model, max_len = num_tokens, gamma = gamma, top_k = top_k, top_p=top_p, random_seed = random_seed)

if __name__ == "__main__":
    #mp.set_start_method('spawn', force=True)
    args = parse_arguments()

    torch.manual_seed(123)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    if args.mode == 'decode':
        generate(args.input, args.approx_model_name, args.target_model_name, num_tokens=args.max_tokens, gamma=args.gamma,
                random_seed = args.seed, verbose=args.verbose, use_benchmark = args.benchmark)

    elif args.mode == 'train_learner':
        #change to use target model name
        target_model = ModelWrapper(args.target_model_name)
        drafter_indices = args.drafters_idx
        if drafter_indices != None:
            drafter_indices = [int(d) for d in drafter_indices]
        #specify the drafters, should change this later
        if drafter_indices != None:
            L = len(drafter_indices)
        else:
            assert False

        #no need for this because apparently the model is already set to the correct devices according to some stack trace that I got
        # target_model.model.to(device)
        # for d in drafters:
        #     d.model.to(device)

        tokenizer = target_model.tokenizer

        #need a dataset, assume that we have some directory called data and some file with one context per line
        # data_file = "data/train.txt"
        # if not os.path.exists(data_file):
        #     raise FileNotFoundError(f"{data_file} not found, please specify a dataset")
        raw_dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
        texts = [item["text"] for item in raw_dataset["train"] if item["text"].strip() != ""]
        #texts = open(data_file, 'r').read().splitlines() #uncomment if we are loading in a dataset
        
        dataset = EnhancedFeatureDataset(tokenizer, target_model, texts, seq_len=128)
        data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn, num_workers=0)

        #create and train Learner then save it afterward with a timestamp
        #learner = LearnerModel(input_dim=4097, hidden_dim=32, L=L).cuda() #llama-7b uses hidden_dim 4096
        #learner = LearnerModel(input_dim=4097, hidden_dim=32, L=L).to(device).half() #bloom-7bm uses hidden_dim 4096
        learner = LearnerModel(input_dim=4097, hidden_dim=32, L=L).to(device)
        #learner.half()

        #accelerator = Accelerator()
        #learner, drafters, target_model, data_loader = accelerator.prepare(learner, drafters, target_model, data_loader)

        epoch_losses = train_learner_with_target(learner, drafter_indices, target_model, data_loader, ptfile=args.ptfile,
                                  metric=args.metric, epochs=args.epochs)

        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filename = f"learner-checkpoints/learner_{timestamp}.pt"
        torch.save(learner.state_dict(), filename)
        print(f"Learner has finished training and the model was saved to {filename}")

        loss_filename = f"learner-checkpoints/learner_losses_{timestamp}.csv"
        with open(loss_filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["epoch", "loss"])
            for epoch_i, loss_val in enumerate(epoch_losses):
                writer.writerow([epoch_i, loss_val])
        print(f"Losses saved to {loss_filename}")
    elif args.mode == 'create_dataset':
        #change to use target model name
        target_model = ModelWrapper(args.target_model_name)

        #specify the drafters, should change this later
        drafters = args.drafters
        drafters = [ModelWrapper(m) for m in drafters]
        L = len(drafters)

        #no need for this because apparently the model is already set to the correct devices according to some stack trace that I got
        # target_model.model.to(device)
        # for d in drafters:
        #     d.model.to(device)

        tokenizer = target_model.tokenizer

        #need a dataset, assume that we have some directory called data and some file with one context per line
        # data_file = "data/train.txt"
        # if not os.path.exists(data_file):
        #     raise FileNotFoundError(f"{data_file} not found, please specify a dataset")
        raw_dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
        texts = [item["text"] for item in raw_dataset["train"] if item["text"].strip() != ""]
        #texts = open(data_file, 'r').read().splitlines() #uncomment if we are loading in a dataset
        
        dataset = EnhancedFeatureDataset(tokenizer, target_model, texts, seq_len=128)
        data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn, num_workers=0)

        #create and train Learner then save it afterward with a timestamp
        #learner = LearnerModel(input_dim=4097, hidden_dim=32, L=L).cuda() #llama-7b uses hidden_dim 4096
        learner = LearnerModel(input_dim=4097, hidden_dim=32, L=L).to(device).half() #bloom-7bm uses hidden_dim 4096
        
        #learner.half()

        #accelerator = Accelerator()
        #learner, drafters, target_model, data_loader = accelerator.prepare(learner, drafters, target_model, data_loader)
        print(args.ptfile)
        sample_training_data(drafters, target_model, data_loader, 
                                  metric=args.metric, epochs=args.epochs, output=args.ptfile)
