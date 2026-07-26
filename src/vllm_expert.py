from transformers import AutoTokenizer
import argparse
from vllm import LLM, SamplingParams
import os
from tqdm import trange
import json
from ipc import init_scheduler_ipc,cleanup_scheduler_ipc
import mmap
import random

# output process id
print(f"[vllm_expert] Process ID: {os.getpid()}")
try:
    ALPHA = float(os.environ["CD_PLAUSIBLE_ALPHA"])
    BETA = float(os.environ["CD_CONTRASTIVE_BETA"])
    JOB_ID = os.environ["JOB_ID"]
except KeyError as e:
    raise RuntimeError(f"❌ Required environment variable not set: {e.args[0]}")
print(f"[vllm_expert] CD_PLAUSIBLE_ALPHA={ALPHA}, CD_CONTRASTIVE_BETA={BETA}")

def main():
    args = parser_args()
    tokenizer_model_id = args.tokenizer_model_id
    expert_model_id = args.expert_model_id
    amateur_model_id = args.amateur_model_id
    jsonl_path = args.jsonl_path
    batch_size = args.batch_size
    max_tokens = args.max_tokens
    temperature = args.temperature
    top_p = args.top_p
    output_path = args.output_path
    use_tqdm = args.use_tqdm_for_each_generation
    tensor_parallel_size = args.tensor_parallel_size
    ban_reasoning_qwen3 = args.ban_reasoning_qwen3
    id_key = args.id_key
    
    if args.seed is not None:
        current_seed = args.seed
    else:
        current_seed = random.randint(0, 1_000_000_000)

    print(f"[vllm_expert] Obtained args: {args}")
    
    #ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_model_id, padding_side="left")
    llm = LLM(model=expert_model_id, dtype="bfloat16", trust_remote_code=True,tensor_parallel_size=tensor_parallel_size,seed=current_seed)
    sp = SamplingParams(max_tokens=max_tokens, temperature=temperature, top_p=top_p,logprobs=0,seed=current_seed)
    
    conversations = []
    conversation_ids = []
    with open(jsonl_path, "r") as f:
        for i, line in enumerate(f):
            obj = json.loads(line)
            instruction = obj.get("instruction")
            conversation_id = obj.get(id_key, f"conv_{i}")
            if instruction:
                conversations.append([
                    {"role": "user", "content": instruction}
                ])
                conversation_ids.append(conversation_id)
    print(f"Loaded {len(conversations)} conversations.")
    print(conversations[:2]) 
    
    # eliminate already processed conversations
    if os.path.exists(output_path):
        unprocessed_conversations = []
        unprocessed_conversation_ids = []
        processed_ids = set()
        with open(output_path, "r") as out_f:
            for line in out_f:
                obj = json.loads(line)
                processed_ids.add(obj.get(id_key))

        for conv, conv_id in zip(conversations, conversation_ids):
            if conv_id not in processed_ids:
                unprocessed_conversations.append(conv)
                unprocessed_conversation_ids.append(conv_id)
        print(f"Continuing from previous runs, {len(unprocessed_conversations)} conversations left to process.")

        conversations = unprocessed_conversations
        conversation_ids = unprocessed_conversation_ids
    
    for i in trange(0,len(conversations), batch_size, desc="Generating batches"):
        batch_convs = conversations[i:i+batch_size]
        batch_conversation_ids = conversation_ids[i:i+batch_size]
        
        if len(batch_convs) < batch_size:
            print(f"[vllm_expert] Last batch size {len(batch_convs)} < {batch_size}, padding to full batch size.")
            pad_conv = [
                    {"role": "user", "content": "Say one word."}
                ]
            pad_conv_id = "pad_conv"
            while len(batch_convs) < batch_size:
                batch_convs.append(pad_conv)
                batch_conversation_ids.append(pad_conv_id)
          
        if ban_reasoning_qwen3:
            prompts = [tokenizer.apply_chat_template(conv, tokenize=False, add_generation_prompt=True,enable_thinking=False)
                   for conv in batch_convs]
        else:
            prompts = [tokenizer.apply_chat_template(conv, tokenize=False, add_generation_prompt=True)
                   for conv in batch_convs]

        outs = llm.generate(prompts, sp,use_tqdm=use_tqdm)
            
        
        with open(output_path, "a") as out_f:
            for j, out in enumerate(outs):
                # skip padded conversations
                if batch_conversation_ids[j] == "pad_conv":
                    continue
                is_completed = out.outputs[0].finish_reason == "stop"
                logprobs = [list(item.values())[0].logprob for item in out.outputs[0].logprobs]
                output_json ={
                    id_key: batch_conversation_ids[j],       
                    "is_completed": is_completed,
                    "instruction": batch_convs[j][0]["content"],
                    "response": out.outputs[0].text,
                    "res_metadata": {
                        "temperature": temperature,
                        "top_p": top_p,
                        "max_tokens": max_tokens,
                        "contrastive_decoding": BETA != 0.0,
                        "alpha": ALPHA,
                        "beta": BETA,
                        "expert_model_id": expert_model_id,
                        "amateur_model_id": amateur_model_id,
                    },
                    "score": logprobs,
                }
                out_f.write(json.dumps(output_json, ensure_ascii=False) + "\n")

    
def parser_args():
    parser = argparse.ArgumentParser(description="vLLM Expert Process")
    parser.add_argument(
        "--expert-model-id",
        type=str,
        default="meta-llama/Llama-3.1-8B-Instruct",
        help="Model ID for vLLM LLM",
    )
    parser.add_argument(
        "--amateur-model-id",
        type=str,
        default="meta-llama/Llama-3.1-8B",
        help="Amateur Model ID for vLLM LLM",
    )
    parser.add_argument(
        "--tokenizer-model-id",
        type=str,
        default="meta-llama/Llama-3.1-8B-Instruct",
        help="Tokenizer Model ID",
    )
    parser.add_argument(
        "--jsonl-path",
        type=str,
        default="../data/example.jsonl",
        help="Path to JSONL file with conversations",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=200,
        help="Maximum number of conversations to load from the JSONL file",
    )
    parser.add_argument(  
        "--max-tokens",
        type=int,
        default=512,
        help="Maximum number of tokens to generate",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Sampling temperature",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=1.0,
        help="Top-p sampling parameter",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default="../outputs/generated_outputs.jsonl",
        help="Path to save generated outputs",
    )
    parser.add_argument(
        "--use-tqdm-for-each-generation",
        action="store_true",
        default=False,
        help="Whether to use tqdm for each generation call",
    )
    parser.add_argument(
        "--tensor_parallel-size",
        type=int,
        default=1,
        help="Tensor parallel size for vLLM LLM",
    )
    parser.add_argument(
        "--ban-reasoning-qwen3",
        action="store_true",
        help="Whether to ban reasoning steps in Qwen-3 models",
    )
    parser.add_argument(
        "--no-ban-reasoning-qwen3",
        dest="ban_reasoning_qwen3",
        action="store_false",
    )
    parser.add_argument("--id_key", type=str, default="id", help="Key to identify unique records")
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Set a fixed seed. If not set, a random seed will be used.",
    )
    return parser.parse_args()

                
if __name__ == "__main__":
    
    if BETA != 0.0:
        # Contrastive Decoding requires IPC setup
        shm, sem_ready, sem_go = init_scheduler_ipc(is_owner=False, JOB_ID=JOB_ID)
        mm = mmap.mmap(shm.fd, shm.size)
        shm.close_fd()
        try:
            main()
        finally:
            cleanup_scheduler_ipc(shm, sem_ready, sem_go, mm, is_owner=True)
    else:
        main()