#!/bin/bash

# Default for optional flags
# If not set (or empty), disable Qwen3 reasoning by default
: "${ban_reasoning_qwen3:=false}"
echo "========================================"
echo " Running with the following configuration"
echo "========================================"
echo " batch_size           = ${batch_size}"
echo " input_path           = ${input_path}"
echo " output_path          = ${output_path_baseline}"
echo " expert_model_id      = ${expert_model_id}"
echo " amateur_model_id     = None (Baseline: no amateur model)"
echo " tensor_parallel_size = ${tensor_parallel_size}"
echo " max_tokens           = ${max_tokens}"
echo " temperature          = ${temperature}"
echo " top_p                = ${top_p}"
echo " ban_reasoning_qwen3  = ${ban_reasoning_qwen3}"
echo " cd_plausible_alpha   = 0 (Baseline: no Contrastive Decoding)"
echo " cd_contrastive_beta  = 0 (Baseline: no Contrastive Decoding)"
echo " CUDA_VISIBLE_DEVICES = ${CUDA_VISIBLE_DEVICES}"
echo "========================================"
echo

now=$(date +"%Y%m%d_%H%M%S")
source ../.venv/bin/activate
# Contrastive Decoding hyperparameters
export CD_PLAUSIBLE_ALPHA=0
export CD_CONTRASTIVE_BETA=0 # Baseline: no Contrastive Decoding
export JOB_ID=$now

# Log directories
log_dir="../logs/baseline/${expert_model_id}"
mkdir -p $log_dir

if [ "$ban_reasoning_qwen3" = "true" ]; then
    BAN_FLAG="--ban-reasoning-qwen3"
else
    BAN_FLAG=""
fi

CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES \
python -u ../src/vllm_expert.py \
    --expert-model-id $expert_model_id \
    --amateur-model-id "" \
    --tokenizer-model-id $expert_model_id \
    --jsonl-path $input_path \
    --output-path $output_path_baseline \
    --batch-size $batch_size \
    --max-tokens $max_tokens \
    --top-p $top_p \
    --tensor_parallel-size $tensor_parallel_size \
    $BAN_FLAG \
    > ${log_dir}/${now}.log 2>&1 
