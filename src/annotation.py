from transformers import AutoTokenizer
from openai import OpenAI
from multiprocessing import Pool
from itertools import repeat
from tqdm import tqdm
import argparse
import gzip
import json
import time
import os
import re
from more_itertools import chunked
from repetition_detector import check_repetition


def run_in_parallel(func, records, processes, desc="", leave=True, total=None):
    """
    processes: number of processes
    records: inputs in tuples
    func: function for multi-processing
    desc: name on process bar
    """
    pool = Pool(processes)
    imap = pool.imap(func, records, chunksize=1)
    if total is None:
        total=len(records)
    result = list(tqdm(imap, ascii=True, desc=desc, leave=leave, total=total))
    pool.close()
    pool.terminate()
    pool.join()
    del pool
    return result


def infer_openai_compatible_api(prompt: str, model: str, t: float, n: int, openai_client, system_message=None, num_tries: int = 0):
    """
    Inference using OpenAI compatible API.

    t: temperature
    n: number of sequences to return
    num_tries: counter for failed number of tries
    openai_client: OpenAI compatible API client instance
    system_message: system prompt (optional)
    """
    try:
        messages = []
        if system_message is not None and len(system_message) > 0:
            messages.append({"role": "system", "content": system_message})
        messages.append({"role": "user", "content": prompt})

        chat_completion = openai_client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=8192,
            temperature=t,
            n=n
        )
    except Exception as e:
        print(f'Encountered Error {e}, trying for the {num_tries} time.')
        time.sleep(10)
        if num_tries >= 3:  # max tries
            return {"text": prompt}
        else:
            return infer_openai_compatible_api(prompt, model, t, n, openai_client, system_message, num_tries=num_tries + 1)

    return chat_completion


def generate_queries_simple(input_dict: dict, template: str, model: str):
    """
    Generate inputs for quality scoring.
    
    input_dict: a dictionary containing instructions and synthesized responses
    template: the template in which to fill in the (instruction, synthesized response) pair
    """
    tokenizer = AutoTokenizer.from_pretrained(model)

    question = input_dict[args.instruction_key]
    answer = input_dict[args.response_key]

    query = template.replace("{question}", question).replace("{answer}", answer)

    answer_tok = tokenizer(answer)
    answer_tok_len = len(answer_tok["input_ids"]) - 1

    return [query], [answer_tok_len], [False]

    
def score_conversation(d: dict, template: str, model: str, t: float, n: int, output_annotation_key: str, system_message=None):
    """
    Main function for scoring (instruction, synthesized response) pair.

    d: a dictionary containing instructions and synthesized responses
    template: the template in which to fill in the (instruction, synthesized response) pair
    model: model name used as LLM-as-a-judge
    t: temperature
    n: number of sequences to return
    openai_client: OpenAI compatible API client instance
    openai_api_key: API Key for OpenAI compatible API
    openai_base_url: Base URL for OpenAI compatible API
    system_message: system prompt (optional)
    """

    # OpenAI compatible API client
    openai_client = OpenAI(
        api_key=args.openai_api_key,
        base_url=args.openai_base_url,
    )
    
    curr_queries, curr_resp_lens, answer_end_ln = generate_queries_simple(d, template, model)
    curr_scoring_annotations = []
    for i, query in enumerate(curr_queries):
        curr_scoring_annotation = {}
        max_retries = 3
        for attempt in range(max_retries):
            completion = infer_openai_compatible_api(query, model, t, n, openai_client, system_message)
            if isinstance(completion, dict):
                # came from retry fallback
                text = completion.get("text", "")
            else:
                text = getattr(completion.choices[0].message, "content", "")
            score = extract_score_from_text(text)

            if score is not None:
                curr_scoring_annotation["judgement"] = text
                break
            else:
                print(f"No score detected, retrying attempt {attempt + 1}/{max_retries}...")
                time.sleep(5)
        else:
            curr_scoring_annotation["judgement"] = text
        curr_scoring_annotation["response_token_length"] = curr_resp_lens[i]
        
        # Complete this code here
        curr_scoring_annotation["repetition_flag"] = check_repetition(d[args.response_key])


        curr_scoring_annotations.append(curr_scoring_annotation)

    d[output_annotation_key] = curr_scoring_annotations
    # d["response_scoring_annotations"] = curr_scoring_annotations
    
    return d

def score_conversation_wrapped(arg):
    return score_conversation(*arg)

import json
import unicodedata
import re

       

def extract_values_from_json(json_string, keys = ["score", "strengths", "weaknesses", "choice"], allow_no_quotes = False):
    extracted_values = {}
    for key in keys:
        if key not in json_string:
            continue
        # Create a regular expression pattern to find the value for the given key
        pattern = f'"{key}"\\s*:\\s*"([^"]*?)"'
        match = re.search(pattern, json_string)
        if match:
            extracted_values[key] = match.group(1)
        else:
            # Handle the case where the value might contain broken quotes
            pattern = f'"{key}"\\s*:\\s*"(.*?)"'
            match = re.search(pattern, json_string, re.DOTALL)
            if match:
                extracted_values[key] = match.group(1)
        if not match and allow_no_quotes:
            # to allow no quotes on the values
            pattern = f'"{key}"\\s*:\\s*([^,\\s]*)'
            match = re.search(pattern, json_string)
            if match:
                extracted_values[key] = match.group(1)
            else:
                # to allow no quotes on the keys
                pattern = f'{key}\\s*:\\s*([^,\\s]*)'
                match = re.search(pattern, json_string)
                if match:
                    extracted_values[key] = match.group(1)
    return extracted_values


def parse_result(result_str, mode="json", eval_mode="score"): 
    assert eval_mode in ["score", "pairwise"]
    try: 
        result_str = result_str.strip() 
        parsed_result = extract_values_from_json(result_str, keys=["score"])
    except Exception as e:
        # print(result_str)
        print(e)
        # raise Exception(f"Failed to parse the result: {result_str}")
        parsed_result = {"score":0}
        # exit()
    return parsed_result


def _to_ascii_digits(s: str) -> str:
    # normalize full-width digits → ASCII
    return "".join(unicodedata.normalize("NFKC", ch) for ch in s)

def extract_score_from_text(judgement: str):
    if args.prompt_name == "wildbench-modified":
        parsed_json = parse_result(judgement)
        if "score" in parsed_json:
            return parsed_json["score"]
        else:
            return None

    if not judgement:
        return None

    text = _to_ascii_digits(judgement).strip()

    # strip code fences if present
    if text.startswith("```"):
        # keep inside the first fenced block if any
        m = re.search(r"```(?:json)?\s*(.+?)\s*```", text, flags=re.S)
        if m:
            text = m.group(1).strip()

    try:
        obj = json.loads(text)
        if isinstance(obj, dict) and "score" in obj:
            val = obj["score"]
            if isinstance(val, (int, float, str)) and str(val).strip():
                return str(val).strip()
    except Exception:
        pass

    m = re.search(r'"score"\s*:\s*([0-9]+(?:\.[0-9])?)', text)
    if m:
        return m.group(1)

    m = re.search(r"\**\[\**\[([0-9]+\.?[0-9]{0,1})\]\**\]\**", text)
    if m:
        return m.group(1)

    m = re.search(r"\**\[([0-9]+\.?[0-9]{0,1})\]\**", text)
    if m:
        return m.group(1)

    matches = re.findall(r"([0-9]+\.?[0-9]{0,1})[/|／]10", text)
    if matches:
        return matches[-1]

    return None



def get_scores(judge_results: list, annotation_key: str):

    """
    Extract scores from judgments of LLM-as-a-judge.
    
    judge_results: a list containing dictionaries of (instruction, synthesized_response, judgment) triples.
    """

    for i, d in tqdm(enumerate(judge_results), desc="Extracting scores..."):
        for completion in d[annotation_key]:

            judgement = completion["judgement"]
            score_str = extract_score_from_text(judgement)            
            try:
                completion["preference_score"] = float(score_str)
            except (TypeError, ValueError):
                completion["preference_score"] = -1
            
            # completion["preference_score"] = eval(score)
            
            if completion["preference_score"] > 100:
                completion["preference_score"] = -1
            elif completion["preference_score"] > 10:
                completion["preference_score"] = completion["preference_score"]/10
    
    return judge_results

    
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", required=True, type=str, help="The path of dataset to process.")
    parser.add_argument("--checkpoint_size", default=1000, type=int, help="Number of samples per checkpoint save")
    parser.add_argument("--num_processes", default=180, type=int, help="Number of processes for multiprocessing")
    parser.add_argument("--model", default="meta-llama/Meta-Llama-3.1-70B-Instruct", type=str, help="Model name for LLM-as-a-judge")
    parser.add_argument("--temperature", default=0.0, type=float, help="Sampling temperature for judgement generation")
    parser.add_argument("--n", default=1, type=int, help="Number of sequences to return per API call")
    parser.add_argument("--openai_api_key", required=True, type=str, help="API key for OpenAI compatible API")
    parser.add_argument("--openai_base_url", required=True, type=str, help="Base URL for OpenAI compatible API")
    parser.add_argument("--raw_output_path", required=True, type=str, help="Raw scoring results output file name (.jsonl)")
    parser.add_argument("--annotated_output_path", required=True, type=str, help="Annotated output file name (.jsonl)")
    parser.add_argument("--prompt_path", default="judge_prompts.jsonl", type=str, help="Prompt definition file (jsonl)")
    parser.add_argument("--prompt_name", default="", type=str, help="Prompt name (name field in prompt file)")
    parser.add_argument("--id_key",default="conversation_id",type=str, help="Key name for id")
    parser.add_argument("--response_key", default="response",type=str, help="Key name for response")
    parser.add_argument("--instruction_key", default="instruction",type=str, help="Key name for instruction")
    parser.add_argument("--annotation_key", default="synthesized_response_scoring_annotations", type=str, help="Key name for scoring annotation field to use (default: `synthesized_response_scoring_annotations`)")
    args = parser.parse_args()
    
    model = args.model
    t = args.temperature
    n = args.n

    ''' check if output files are already exist '''
    processed_ids = set()
    if os.path.exists(args.raw_output_path):
        with open(args.raw_output_path, "r", encoding="utf-8") as f:
            for line in f:
                d = json.loads(line)
                if args.annotation_key in d:
                    processed_ids.add(d.get(args.id_key))
    
    ''' load data. '''
    dataset = []
    with open(args.dataset_path, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc="Reading File"):
            d = json.loads(line)
            if d.get(args.id_key) in processed_ids:
                continue
            dataset.append(d)
            
    ''' load prompt. '''
    prompt_template = None
    system_message = None
    with open(args.prompt_path) as f:
        for line in f:
            prompt = json.loads(line)
            if prompt.get("name") == args.prompt_name:
                prompt_template = prompt["prompt_template"]
                system_message = prompt.get("system_prompt", None)
                break
    if prompt_template is None:
        raise ValueError("Prompt template not found for the specified prompt_name.")
    if system_message is not None:
        Warning(f"Following system message will be used: {system_message}")
    

    ''' run inference using LLM-as-a-judge. '''
    print(f"Total samples to process: {len(dataset)}")
    checkpoint_size = args.checkpoint_size

    for chunk in chunked(dataset, checkpoint_size):
        dataset_with_judgements = run_in_parallel(
            func=score_conversation_wrapped,
            records=tuple(zip(
                chunk,
                repeat(prompt_template),
                repeat(model),
                repeat(t),
                repeat(n),
                repeat(args.annotation_key),
                repeat(system_message)
            )),
            processes=args.num_processes,
            desc="Judging responses"
        )

        # raw
        with open(args.raw_output_path, "a", encoding="utf-8") as wf:
            for d in dataset_with_judgements:
                wf.write(json.dumps(d, ensure_ascii=False) + "\n")

        # score
        dataset_with_scores = get_scores(
            dataset_with_judgements,
            args.annotation_key
        )

        # annotated
        with open(args.annotated_output_path, "a", encoding="utf-8") as wf:
            for d in dataset_with_scores:
                wf.write(json.dumps(d, ensure_ascii=False) + "\n")

        print(f"Checkpoint saved: {len(dataset_with_judgements)} samples")