#!/bin/bash
cd "$(dirname "$0")" || exit 1

# Use Contrastive Decoding or Not
USE_CONTRASTIVE_DECODING=${1:-"true"}

# Model settings
tensor_parallel_size=${2:-1} # If you use Contrastive Decoding, need 2*tensor_parallel_size gpus.
expert_model_id="meta-llama/Llama-3.2-1B-Instruct"
amateur_model_id="meta-llama/Llama-3.2-1B"

# Dataset
input_path=${3:-"../data/example.jsonl"}
output_path=${4:-"../outputs/example-Llama-3.2-1B-CD-test.jsonl"}
output_path_baseline=${5:-"../outputs/example-Llama-3.2-1B.jsonl"}

# generation parameters
max_tokens=4096
temperature=1.0
top_p=1.0
batch_size=200
ban_reasoning_qwen3=false


cd_contrastive_beta=1
cd_plausible_alpha=${6:-0.01}

# GPU selection
CUDA_VISIBLE_DEVICES_AMA="0"
CUDA_VISIBLE_DEVICES_EXP="1"
CUDA_VISIBLE_DEVICES="0,1" # Only for baseline script

export \
  batch_size \
  input_path \
  output_path \
  output_path_baseline \
  expert_model_id \
  amateur_model_id \
  tensor_parallel_size \
  max_tokens \
  temperature \
  ban_reasoning_qwen3 \
  top_p \
  cd_plausible_alpha \
  cd_contrastive_beta \
  CUDA_VISIBLE_DEVICES \
  CUDA_VISIBLE_DEVICES_AMA \
  CUDA_VISIBLE_DEVICES_EXP \

if [ "$USE_CONTRASTIVE_DECODING" = "true" ]; then
  bash run_common.sh
else
  bash run_baseline_response.sh
fi