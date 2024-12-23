#!/bin/bash
#SBATCH --job-name=learner-distill-pythia
#SBATCH --account=kempner_emalach_lab
#SBATCH --output=/n/holylfs06/LABS/kempner_fellow_emalach/Lab/rli/2281-Project/logs-distillation/%j/logs.out
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

nvidia-smi

#project directory
cd /n/holylfs06/LABS/kempner_fellow_emalach/Lab/rli/2281-Project

#environment variables
#export SPECIAL_TOKEN="<|sep|>"

#run the script
python main.py --mode distill \
    --dataset_name wikitext \
    --dataset_config wikitext-2-raw-v1 \
    --dataset_split train \
    --distillation_directory distillation \
    --target_model_name bigscience/bloom-7b1 \
    --student_model_name bigscience/bloom-1b1 \
    --epochs=20 \
    --batch_size=4 \
    --lr_distillation=1e-5 \
    --temperature=2.0
    
#     &
# PYTHON_PID=$!

# #print nvidia-smi ever so often
# while kill -0 $PYTHON_PID 2>/dev/null; do
#     nvidia-smi
#     sleep 420
# done
