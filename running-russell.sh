#!/bin/bash
#SBATCH --job-name=bloom-dataset
#SBATCH --account=kempner_emalach_lab
#SBATCH --output=/n/holylfs06/LABS/kempner_fellow_emalach/Lab/rli/2281-Project/logs/%j/logs.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=russell_li@college.harvard.edu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=16:00:00
#SBATCH --mem=250GB
#SBATCH --partition=kempner_h100
#SBATCH --constraint=h100
#SBATCH --exclude=holygpu8a19405
#SBATCH --array=1-1

#load modules
module load python/3.10.9-fasrc01
module load cuda/11.8.0-fasrc01 cudnn/8.9.2.26_cuda11-fasrc01

#activate conda environment
source ~/.bashrc
conda activate 2281-project-env

#project directory
cd /n/holylfs06/LABS/kempner_fellow_emalach/Lab/rli/2281-Project

#environment variables
#export SPECIAL_TOKEN="<|sep|>"

#run the finetuning script
python main.py --mode train_learner \
    --target_model_name EleutherAI/pythia-2.8b \
    --ptfile data/pythia-lk-epochs20.pt \
    --drafters_idx 0 1 2 \
    --metric=lk \
    --lk_k=1 \
    --epochs=20 \
    --hidden_dim=512 \
    --num_layers=25 \
    --dropout=0.5

#python main.py --target_model_name bigscience/bloomz-7b1 --mode create_dataset --epochs 20 --batch_size 16 --drafters bigscience/bloomz-560m bigscience/bloom-560m bigscience/bloomz-1b1 bigscience/bloom-1b1 --metric 'lk' --ptfile 'bloom-lk-epochs20.pt' --sizes 56 56 110 110
