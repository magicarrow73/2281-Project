#!/bin/bash
#SBATCH --job-name=learner-sweep-bloom
#SBATCH --account=kempner_emalach_lab
#SBATCH --output=/n/holylfs06/LABS/kempner_fellow_emalach/Lab/rli/2281-Project/logs-bloom-sweep-batch4/%j/logs.out
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
#SBATCH --array=1-45%9

#load modules
module load python/3.10.9-fasrc01
module load cuda/11.8.0-fasrc01 cudnn/8.9.2.26_cuda11-fasrc01

#activate conda environment
source ~/.bashrc
conda activate 2281-project-env

#project directory
cd /n/holylfs06/LABS/kempner_fellow_emalach/Lab/rli/2281-Project

#environment variables
wandb login $WANDB_API_KEY

TARGET_MODEL="bigscience/bloomz-7b1"
PTFILE="data/bloom-lk-epochs20-batch4.pt"
METRIC="lk"
LK_K=1
EPOCHS=20
DROPOUT=0.3
MODE="train_learner"
WANDB_PROJECT="2281-project"

CHECKPOINT_DIR="learner-checkpoints-bloom"

PARAMS_LINE=$(sed -n ${SLURM_ARRAY_TASK_ID}p params-sweeping/params-bloom.txt)
read -r DRAFTERS_IDX_STRING HDIM LAYERS <<< "$PARAMS_LINE"
DRAFTERS_IDX=$(echo $DRAFTERS_IDX_STRING | tr '-' ' ')
DRAFTERS_IDX_NAME=$(echo $DRAFTERS_IDX | tr ' ' '_')
RUN_NAME="drafters_${DRAFTERS_IDX_NAME}_hdim_${HDIM}_layers_${LAYERS}"

echo "Running $RUN_NAME with drafters_idx='$DRAFTERS_IDX', hidden_dim=$HDIM, num_layers=$LAYERS"

SIZES_STR=""
for idx in $DRAFTERS_IDX; do
    if [ "$idx" -eq 0 ]; then
        SIZES_STR="$SIZES_STR 56"
    elif [ "$idx" -eq 1 ]; then
        SIZES_STR="$SIZES_STR 56"
    elif [ "$idx" -eq 2 ]; then
        SIZES_STR="$SIZES_STR 110"
    elif [ "$idx" -eq 3 ]; then
        SIZES_STR="$SIZES_STR 110"
    else
        echo "Unknown drafter index $idx"
        exit 1
    fi
done

SIZES_STR=$(echo $SIZES_STR | sed 's/^ *//g')

python main.py --mode $MODE \
    --target_model_name $TARGET_MODEL \
    --ptfile $PTFILE \
    --drafters_idx $DRAFTERS_IDX \
    --metric=$METRIC \
    --lk_k=$LK_K \
    --epochs=$EPOCHS \
    --hidden_dim=$HDIM \
    --num_layers=$LAYERS \
    --dropout=$DROPOUT \
    --sizes $SIZES_STR \
    --save_interval=100 \
    --batch_size=4 \
    --wandb_project=$WANDB_PROJECT \
    --wandb_run_name=$RUN_NAME \
    --checkpoint_dir=$CHECKPOINT_DIR
