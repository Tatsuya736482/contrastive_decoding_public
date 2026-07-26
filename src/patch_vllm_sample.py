# This file overrides the Sampler.forward method in vLLM v0.11.0 
# ref: https://github.com/vllm-project/vllm/blob/v0.11.0/vllm/v1/sample/sampler.py#L69

# vllm/vllm/entrypoints/llm.py
import os
import inspect
import sys
from vllm.v1.outputs import SamplerOutput
import torch
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.entrypoints.llm import LLM
import json, zmq, numpy as np, torch, zlib, os


import threading, zmq, os, json, numpy as np, torch, zlib
import vllm
from vllm.v1.sample.sampler import Sampler
import time
from dotenv import load_dotenv, find_dotenv
from coordinator import Kind
from transformers import AutoTokenizer

try:
    ALPHA = float(os.environ["CD_PLAUSIBLE_ALPHA"])
    BETA = float(os.environ["CD_CONTRASTIVE_BETA"])
    TOKENIZER_MODEL_ID = os.environ["TOKENIZER_MODEL_ID"]
    SYNC_ADDRS = os.getenv("VLLM_SYNC_ADDRS", "").split(",")
    _SYNC_DTYPE = os.environ.get("VLLM_SYNC_DTYPE")
except KeyError as e:
    raise RuntimeError(f"❌ Required environment variable not set: {e.args[0]}")


_ctx = None
_sock_map = [{} for _ in range(len(SYNC_ADDRS))]   # kind -> zmq.Socket
_lock_map = [{} for _ in range(len(SYNC_ADDRS))]  # kind -> threading.Lock
def _ctx_inst():
    global _ctx
    if _ctx is None:
        _ctx = zmq.Context.instance()
    return _ctx

def _sock_for(kind: str,device: torch.device):
    cuda_idx = device.index
    if cuda_idx >= len(SYNC_ADDRS):
        raise RuntimeError(f"❌ CUDA device index {cuda_idx} out of range for SYNC_ADDRS with length {len(SYNC_ADDRS)}")
    s = _sock_map[cuda_idx].get(kind, None)
    if s is None:
        s = _ctx_inst().socket(zmq.REQ)
        s.connect(SYNC_ADDRS[cuda_idx])
        s.setsockopt(zmq.RCVTIMEO, 100000)  # 100 seconds
        s.setsockopt(zmq.SNDTIMEO, 100000) # 100 seconds
        _sock_map[cuda_idx][kind] = s
        _lock_map[cuda_idx][kind] = threading.Lock()
    return s, _lock_map[cuda_idx][kind]

def _sync_call(kind: str, parts_out: list[bytes],device:torch.device) -> list[bytes]:
    s, lk = _sock_for(kind,device)
    with lk:
        s.send_multipart(parts_out)
        parts_in = s.recv_multipart()
    return parts_in

def structure_to_int(x, step: int):
    """Returns a hash integer representing the exact structure of nested lists and a step number."""
    def shape(obj):
        if isinstance(obj, list):
            return [len(obj)] + [shape(item) for item in obj]
        else:
            return [0]
    
    data = {"shape": shape(x), "step": step}
    s = json.dumps(data, sort_keys=True)
    return zlib.adler32(s.encode("utf-8")) & 0xFFFFFFFF


def _send_logprobs_recv_sampled(t: torch.Tensor, hash:int, device:torch.device) -> torch.Tensor:
    if t.ndim == 1:
        t = t.unsqueeze(0)

    if _SYNC_DTYPE == "f16":
        t_cpu = t.detach().to(torch.float16).contiguous().cpu(); dtype_tag = "f16"
    else:
        t_cpu = t.detach().to(torch.float32).contiguous().cpu(); dtype_tag = "f32"

    B, V = t.shape
    header = {
        "kind": Kind.SEND_LOGPROBS_RECV_SAMPLED.value,
        "dtype": dtype_tag,
        "shape": [B, V],
        "hash": hash,
    }
    header_b = json.dumps(header).encode("utf-8")
    payload = memoryview(t_cpu.numpy())

    rep_h_b, rep_p_b = _sync_call(Kind.SEND_LOGPROBS_RECV_SAMPLED.value, [header_b, payload],device)
    rep = json.loads(rep_h_b.decode("utf-8"))
    rep_dtype = rep.get("dtype", "i32")
    rep_shape = rep.get("shape", None)
    

    assert rep.get("kind") == Kind.SEND_LOGPROBS_RECV_SAMPLED.value, f"kind mismatch: expected {Kind.SEND_LOGPROBS_RECV_SAMPLED.value}, got {rep.get('kind')}"
    assert rep.get("hash") == hash, f"hash mismatch: expected {hash}, got {rep.get('hash')}"
    assert rep_shape is not None, "missing shape for sampled tokens"
    assert rep_dtype in ("i32", "i64"), f"unexpected token dtype: {rep_dtype}"
    np_dtype = np.int32 if rep_dtype == "i32" else np.int64
    arr = np.frombuffer(rep_p_b, dtype=np_dtype).copy()
    
    assert tuple(rep_shape) == (arr.shape[0],), f"shape mismatch: header {rep_shape}, buffer {arr.shape}"

    return torch.from_numpy(arr.astype(np.int64, copy=False))


def _send_waiting_recv_logits(hash:int, device:torch.device) -> torch.Tensor:
    header = {"kind": Kind.SEND_WAITING_RECV_LOGITS.value, "hash": hash}
    header_b = json.dumps(header).encode("utf-8")

    rep_h_b, rep_p_b = _sync_call(Kind.SEND_WAITING_RECV_LOGITS.value, [header_b, b""],device)
    rep = json.loads(rep_h_b.decode("utf-8"))

    kind = rep.get("kind")
    dtype_tag = rep.get("dtype")
    shape = rep.get("shape")
    hash_recv = rep.get("hash")
    
    assert kind == Kind.SEND_WAITING_RECV_LOGITS.value, f"kind mismatch: expected {Kind.SEND_WAITING_RECV_LOGITS.value}, got {kind}"
    assert dtype_tag in ("f16", "f32"), f"invalid dtype: {dtype_tag}"
    assert shape is not None, "shape is None"
    assert hash == hash_recv, f"hash mismatch: expected {hash}, got {hash_recv}"
        
    np_dtype = np.float16 if dtype_tag == "f16" else np.float32
    arr = np.frombuffer(rep_p_b, dtype=np_dtype).reshape(shape)
    t = torch.from_numpy(arr)
    return t.to(device=device)

def _send_sampled_tokens_recv_ack(token_ids: torch.Tensor, hash:int,device:torch.device) -> bool:
    if token_ids.ndim != 1:
        token_ids = token_ids.reshape(-1)
    
    ids_cpu = token_ids.detach().to(torch.int32).contiguous().cpu().numpy()
    payload = memoryview(ids_cpu)
    header = {
        "kind": Kind.SEND_SAMPLED_RECV_ACK.value,
        "dtype": "i32",
        "shape": [int(ids_cpu.shape[0])],
        "hash": hash
    }
    header_b = json.dumps(header).encode("utf-8")

    [rep_h_b] = _sync_call(Kind.SEND_SAMPLED_RECV_ACK.value, [header_b, payload],device)
    rep = json.loads(rep_h_b.decode("utf-8"))
    hash_recv = rep.get("hash")
    kind = rep.get("kind")

    assert hash == hash_recv, f"hash mismatch: expected {hash}, got {hash_recv}"
    assert kind == Kind.SEND_SAMPLED_RECV_ACK.value, f"kind mismatch: expected {Kind.SEND_SAMPLED_RECV_ACK.value}, got {kind}"
    return True


def plausibleLogitsWarper(log_probs, alpha, filter_value=-float("Inf")):
    probs = log_probs.exp()
    probs_max = probs.max(axis=-1, keepdim=True).values
    indices_to_remove = (probs < alpha * probs_max)
    log_probs = log_probs.masked_fill(indices_to_remove, filter_value)
    return log_probs


def install_patch(role="Expert"):
    if role not in ("Expert", "Amateur"):
        raise ValueError("role must be 'Expert' or 'Amateur'")
    VLLM_MIN = (0, 11, 0) # vLLM minimum version required for this patch
    
    # check vLLM version
    ver = tuple(int(x) for x in getattr(vllm, "__version__", "0.0.0").split(".")[:3])
    if ver < VLLM_MIN:
        print(f"[patch] vLLM {ver} < min {VLLM_MIN}; patch may not match", file=sys.stderr)

    # patch Sampler.forward  
    step = 0
    max_batch_size = None
    debug_cd_flag = True
    orig = Sampler.forward
    
    # tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_MODEL_ID)

    def forward_expert(
            self,
            logits: torch.Tensor,
            sampling_metadata: SamplingMetadata
        ) -> SamplerOutput:
        nonlocal step
        nonlocal max_batch_size
        nonlocal debug_cd_flag
        
        # compute hash from step number and structure of output_token_ids to synchronize with amateur
        step += 1
        hash = structure_to_int(sampling_metadata.output_token_ids, step)
        
        max_batch_size = max(max_batch_size or 0, logits.shape[0])
        is_all_empty = all(sampling_metadata.output_token_ids[i] == [] for i in range(logits.shape[0]))
        # print(step,logits.shape)
        if logits.shape[0] == max_batch_size and is_all_empty:
            print(f"😀[patch] Expert samling is not synced with Amateur at step {step}, skipping sync.")
            sampler_output = orig(self,logits,sampling_metadata)
            return sampler_output

        raw_logprobs_exp = self.compute_logprobs(logits)
        
        raw_logprobs_ama = _send_waiting_recv_logits(
            device=logits.device,
            hash=hash
        )

        raw_logprobs_exp = plausibleLogitsWarper(
            raw_logprobs_exp,
            alpha=ALPHA,
            filter_value=-float("Inf")
        )

        # check the size matches
        if raw_logprobs_ama.shape != raw_logprobs_exp.shape:
            raise RuntimeError(f"shape mismatch: expert {raw_logprobs_exp.shape}, amateur {raw_logprobs_ama.shape}")
        
        logits_new =  raw_logprobs_exp - BETA * raw_logprobs_ama

        if debug_cd_flag:
            print("\n" + "=" * 70)
            print(f"🎉  [PATCH] Contrastive Decoding Enabled {ALPHA}, {BETA} 🎉")
            print("=" * 70 + "\n")
            debug_cd_flag = False
        
        sampler_output =  orig(self,logits_new,sampling_metadata)

        try:
            res = _send_sampled_tokens_recv_ack(
                sampler_output.sampled_token_ids,
                hash=hash,
                device=logits.device
            )
            if not res:
                raise RuntimeError("Expert: ACK verification failed")
        except Exception as e:
            raise e
        
        return sampler_output
            
    def forward_amateur(
            self,
            logits: torch.Tensor,
            sampling_metadata: SamplingMetadata
        ) -> SamplerOutput:
        nonlocal step
        nonlocal max_batch_size
        # print(f"😀{step}: {logits.shape} {logits.device} first 10 logits: {logits[:10]}")
        # compute hash from step number and structure of output_token_ids to synchronize with expert
        step += 1
        hash = structure_to_int(sampling_metadata.output_token_ids, step)
        
        max_batch_size = max(max_batch_size or 0, logits.shape[0])
        is_all_empty = all(sampling_metadata.output_token_ids[i] == [] for i in range(logits.shape[0]))
        if logits.shape[0] == max_batch_size and is_all_empty:
            print(f"😀[patch] Amateur samling is not synced with Expert at step {step}, skipping sync.")
            sampler_output = orig(self,logits,sampling_metadata)
            return sampler_output

        raw_logprobs = self.compute_logprobs(logits) 
       
        try:
            sampler_output = _send_logprobs_recv_sampled(
                t=raw_logprobs,
                hash=hash,
                device=logits.device
            ) 
        except Exception as e:
            raise e


        sampler_output = SamplerOutput(
            # The sampled tokens are expanded to 2D tensor with shape
            # [num_requests, 1], where each row represents one generated
            # token per request.
            sampled_token_ids=sampler_output.unsqueeze(-1).to(device=logits.device, dtype=torch.long),
            logprobs_tensors=None,
        )

        return sampler_output
        
    
    if role == "Expert":
        print(f"😀[patch] Patched Expert Sampler")
        Sampler.forward = forward_expert
    elif role == "Amateur":
        print(f"😀[patch] Patched Amateur Sampler")
        Sampler.forward = forward_amateur
    else:
        raise ValueError("role must be 'Expert' or 'Amateur'")
