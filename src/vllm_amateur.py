from transformers import AutoTokenizer, AutoConfig
import json
from vllm import LLM, SamplingParams
import os
from tqdm import trange
from vllm_expert import parser_args
import json
from transformers import GenerationConfig
import random
# output process id
print(f"[vllm_amateur] Process ID: {os.getpid()}")

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

    print(f"[vllm_amateur] Obtained args: {args}") 
    
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_model_id, padding_side="left")
    
    gen_cfg = GenerationConfig.from_pretrained(tokenizer_model_id)
    eos_ids = gen_cfg.eos_token_id

    if isinstance(eos_ids, int):
        stop_ids = [eos_ids]
    elif isinstance(eos_ids, (list, tuple)):
        stop_ids = list(eos_ids)
    else:
        raise ValueError("Unexpected eos_token_id type in config.json")
    print(f"[vllm_amateur] Using stop_token_ids={stop_ids} for generation.")
    
    llm = LLM(model=amateur_model_id, dtype="bfloat16", trust_remote_code=True,tokenizer=tokenizer_model_id,tensor_parallel_size=tensor_parallel_size,seed=current_seed)
    sp = SamplingParams(max_tokens=max_tokens, temperature=temperature, top_p=top_p, stop_token_ids=stop_ids,seed=current_seed)
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
                
        if len(batch_convs) < batch_size:
            print(f"[vllm_amateur] Last batch size {len(batch_convs)} < {batch_size}, padding to full batch size.")
            pad_conv = [
                    {"role": "user", "content": "Say one word."}
                ]
            while len(batch_convs) < batch_size:
                batch_convs.append(pad_conv)
          
        if ban_reasoning_qwen3:
            prompts = [tokenizer.apply_chat_template(conv, tokenize=False, add_generation_prompt=True,enable_thinking=False)
                   for conv in batch_convs]
        else:
            prompts = [tokenizer.apply_chat_template(conv, tokenize=False, add_generation_prompt=True)
                   for conv in batch_convs]

        llm.generate(prompts, sp,use_tqdm=use_tqdm)
        
        # expert process saves outputs, no need to save again in amateur process
       
if __name__ == "__main__":
    main()