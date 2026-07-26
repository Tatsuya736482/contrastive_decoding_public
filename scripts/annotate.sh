#!/bin/bash
set -e

cd "$(dirname "$0")" || exit 1
source ../.venv/bin/activate
source ../.env
JSONL_PATH=${1:-"../data/lmsys-examples.jsonl"}
PROMPT_NAME=${2:-"wildbench-modified"}

# path manipulation
DIRNAME=$(dirname "$JSONL_PATH")
BASENAME=$(basename "$JSONL_PATH" .jsonl)

RAW_PATH="${DIRNAME}/${BASENAME}.${PROMPT_NAME}.raw.jsonl"
ANNOTATED_PATH="${DIRNAME}/${BASENAME}.${PROMPT_NAME}.annotated.jsonl"

echo "Input      : $JSONL_PATH"
echo "Raw output : $RAW_PATH"
echo "Annotated  : $ANNOTATED_PATH"

URL="https://huggingface.co/api/models/openai/gpt-oss-120b"
THRESHOLD=100
MAX_RETRIES=3


for (( i=1; i<=MAX_RETRIES; i++ )); do
    echo "========================================"
    echo "Checking Rate Limit to: $URL"
    echo "Check Attempt: $i / $MAX_RETRIES"
    HEADERS=$(curl -s -I "$URL" -H "Authorization: Bearer $HF_TOKEN" | tr -d '\r')
    REMAINING=$(echo "$HEADERS" | grep -i "ratelimit:" | grep -o 'r=[0-9]*' | cut -d= -f2)
    RESET_SEC=$(echo "$HEADERS" | grep -i "ratelimit:" | grep -o 't=[0-9]*' | cut -d= -f2)

    if [ -z "$REMAINING" ]; then
        echo "⚠️  Couldn't get rate limit info. Headers might be missing."
        REMAINING=0
        RESET_SEC=10
    fi

    echo "Current Remaining: $REMAINING requests"
    echo "Time to Reset: $RESET_SEC seconds"

    if [ "$REMAINING" -gt "$THRESHOLD" ]; then
        echo "✅ Quota is sufficient ($REMAINING > $THRESHOLD)."
        echo "Exiting check loop."
        break
    else
        echo "🛑 Remaining quota is below $THRESHOLD."

        if [ "$i" -eq "$MAX_RETRIES" ]; then
            echo "❌ Maximum retries reached. Still rate limited. Exiting..."
            exit 1
        fi
        echo "Waiting $RESET_SEC seconds for limit reset..."
        sleep $(($RESET_SEC + 1))
        echo "Wait complete! Retrying check..."
    fi
done

echo "----------------------------------------"
echo "🚀 Resuming main process..."


# vLLM
vllm serve openai/gpt-oss-120b \
  --tensor-parallel-size 2 \
  --async-scheduling \
  --host 0.0.0.0 \
  --port 8000 &

VLLM_PID=$!

# wait for vLLM to be ready
until curl -s http://localhost:8000/v1/models > /dev/null; do
  sleep 1
done

echo "vLLM is ready!"

# annotation 実行
python ../src/annotation.py \
  --dataset_path "$JSONL_PATH" \
  --num_processes 128 \
  --model openai/gpt-oss-120b \
  --temperature 1.0 \
  --n 1 \
  --checkpoint_size 5000 \
  --id_key id \
  --openai_api_key dummy \
  --openai_base_url http://localhost:8000/v1 \
  --raw_output_path "$RAW_PATH" \
  --annotated_output_path "$ANNOTATED_PATH" \
  --prompt_path prompts/judge_prompts.jsonl \
  --prompt_name $PROMPT_NAME

# shutdown vLLM
echo "Stopping vLLM..."
kill $VLLM_PID
