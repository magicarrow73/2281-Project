#!/usr/bin/env bash

set -e

ENV_NAME=2281-project-env
REQUIREMENTS_FILE=requirements.txt

mamba create -n "$ENV_NAME" python=3.9 -y
mamba activate "$ENV_NAME"

pip install -r "$REQUIREMENTS_FILE"
pip install accelerate
pip install --upgrade --force-reinstall "numpy<2.0"
pip install --upgrade torch --index-url https://download.pytorch.org/whl/cu118
pip install sentencepiece
pip install bitsandbytes
pip install datasets

#huggingface token for llama
pip install huggingface_hub
read -sp "put your hf token here: " HF_TOKEN
echo

huggingface-cli login --token $HF_TOKEN

echo "finished setting up conda env"
