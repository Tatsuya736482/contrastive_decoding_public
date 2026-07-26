#!/bin/bash

# Default for optional flags
# If not set (or empty), disable Qwen3 reasoning by default
: "${ban_reasoning_qwen3:=false}"
echo "========================================"
echo " Running with the following configuration"
echo "========================================"
echo " batch_size           = ${batch_size}"
echo " input_path           = ${input_path}"
echo " output_path          = ${output_path}"
echo " expert_model_id      = ${expert_model_id}"
echo " amateur_model_id     = ${amateur_model_id}"
echo " tensor_parallel_size = ${tensor_parallel_size}"
echo " max_tokens           = ${max_tokens}"
echo " temperature          = ${temperature}"
echo " top_p                = ${top_p}"
echo " ban_reasoning_qwen3  = ${ban_reasoning_qwen3}"
echo " cd_plausible_alpha   = ${cd_plausible_alpha}"
echo " cd_contrastive_beta  = ${cd_contrastive_beta}"
echo " CUDA_VISIBLE_DEVICES_AMA  = ${CUDA_VISIBLE_DEVICES_AMA}"
echo " CUDA_VISIBLE_DEVICES_EXP  = ${CUDA_VISIBLE_DEVICES_EXP}"
echo "========================================"
echo

now=$(date +"%Y%m%d_%H%M%S")
source ../.venv/bin/activate
# Contrastive Decoding hyperparameters
export CD_PLAUSIBLE_ALPHA=$cd_plausible_alpha
export CD_CONTRASTIVE_BETA=$cd_contrastive_beta
export TOKENIZER_MODEL_ID=$expert_model_id
export JOB_ID=$now

# Log directories
coord_log_dir="../logs/coord/${expert_model_id}"
amateur_log_dir="../logs/amateur/${expert_model_id}"
expert_log_dir="../logs/expert/${expert_model_id}"
mkdir -p $coord_log_dir
mkdir -p $amateur_log_dir
mkdir -p $expert_log_dir

# Trap to cleanup background processes on exit
cleanup() {
  kill -TERM "$pid_expert" "$pid_amateur" "$pid_coord" 2>/dev/null || true
}
trap 'cleanup' EXIT HUP INT TERM

# coodinator server to sync between expert and amateur
port_start=10000
port_end=11000
used_ports=()
sync_addrs=()

for ((i=0; i<tensor_parallel_size; i++)); do
  for port in $(seq $port_start $port_end); do
    (echo > /dev/tcp/127.0.0.1/$port) >/dev/null 2>&1
    if [ $? -ne 0 ]; then
      if [[ " ${used_ports[@]} " =~ " $port " ]]; then
        continue
      fi

      used_ports+=($port)
      addr="tcp://127.0.0.1:$port"
      sync_addrs+=("$addr")

      export VLLM_SYNC_ADDR_BIND="tcp://*:$port"
      echo "[$i] 🚀Starting coordinator server at $addr"
      python -u ../src/coordinator.py > ${coord_log_dir}/${now}_$i.log 2>&1 &
      pid_coord=$!
      echo "[$i] ✅Started process with PID: $pid_coord (port $port)"
      break
    fi
  done
done

export VLLM_SYNC_DTYPE="f16"
export VLLM_SYNC_ADDRS=$(IFS=,; echo "${sync_addrs[*]}")
echo "🌐 All sync addresses: $VLLM_SYNC_ADDRS"

export VLLM_BATCH_SIZE=$batch_size
export VLLM_NO_USAGE_STATS=1

# Amateur process
echo "🚀Starting Amateur on GPU: $CUDA_VISIBLE_DEVICES_AMA"


if [ "$ban_reasoning_qwen3" = "true" ]; then
    BAN_FLAG="--ban-reasoning-qwen3"
else
    BAN_FLAG=""
fi


CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES_AMA \
PYTHONPATH=$(pwd)/../src/sitecustomize/Amateur:$PYTHONPATH \
python -u ../src/vllm_amateur.py \
    --expert-model-id $expert_model_id \
    --amateur-model-id $amateur_model_id \
    --tokenizer-model-id $expert_model_id \
    --jsonl-path $input_path \
    --output-path $output_path \
    --batch-size $batch_size \
    --max-tokens $max_tokens \
    --temperature $temperature \
    --top-p $top_p \
    --tensor_parallel-size $tensor_parallel_size \
    $BAN_FLAG \
    > ${amateur_log_dir}/${now}.log 2>&1 &
pid_amateur=$!       
echo "✅Started process with PID: $pid_amateur"

# Expert process
echo "🚀Starting Expert on GPU: $CUDA_VISIBLE_DEVICES_EXP "
CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES_EXP \
PYTHONPATH=$(pwd)/../src/sitecustomize/Expert:$PYTHONPATH \
python -u ../src/vllm_expert.py \
    --expert-model-id $expert_model_id \
    --amateur-model-id $amateur_model_id \
    --tokenizer-model-id $expert_model_id \
    --jsonl-path $input_path \
    --output-path $output_path \
    --batch-size $batch_size \
    --max-tokens $max_tokens \
    --top-p $top_p \
    --tensor_parallel-size $tensor_parallel_size \
    $BAN_FLAG \
    > ${expert_log_dir}/${now}.log 2>&1 &
pid_expert=$!
echo "✅Started process with PID: $pid_expert"

wait -n "$pid_expert" "$pid_amateur" "$pid_coord" || true