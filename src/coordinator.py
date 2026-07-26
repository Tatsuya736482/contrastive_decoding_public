import os, json, time, zlib, numpy as np, zmq
from dataclasses import dataclass, field
from typing import Optional, Dict, Tuple
from enum import Enum

try:
    BIND_ADDR = os.environ.get("VLLM_SYNC_ADDR_BIND")
except KeyError as e:
    raise RuntimeError(f"❌ Required environment variable not set: {e.args[0]}")

# print this process id
print(f"{BIND_ADDR}🚀[coord] Coordinator PID: {os.getpid()}")
class Kind(Enum):
    SEND_LOGPROBS_RECV_SAMPLED = "send_logprobs_recv_sampled"
    SEND_WAITING_RECV_LOGITS = "send_waiting_recv_logits"
    SEND_SAMPLED_RECV_ACK = "send_sampled_recv_ack"

class Role(Enum):
    EXPERT = "expert"
    AMATEUR = "amateur"
    
@dataclass
class Logits:
    logits : np.ndarray
    dtype : str
    shape : Tuple[int]
    structure_hash : int = -1
    
@dataclass
class WaitingState:
    hash: int = -1
    waiting_kind: Kind = None
    identity: Optional[bytes] = None
    identity_role: Role = None      # "expert" or "amateur"
    mode: str = "REQ"                 # "REQ" or "DEALER"
    ama_logits : Optional[Logits] = field(default_factory=lambda: Logits(logits=b"", dtype="", shape=()))


peer_mode = {}
def parse_incoming(parts):
    identity = parts[0]
    idx = 1
    if len(parts) > 1 and parts[1] == b'':
        peer_mode[identity] = "REQ"
        idx = 2
    else:
        peer_mode.setdefault(identity, "DEALER")
    header_b = parts[idx] if len(parts) > idx else b'{}'
    payload_b = parts[idx + 1] if len(parts) > idx + 1 else b''
    return identity, header_b, payload_b
    
def commit_sampled_tokens(identity, header, payload_b, sock, waiting_state):
    identity_exp = identity
    hash_expert = header.get("hash", -1)
    dtype = header.get("dtype", "i32")
    shape = header.get("shape", [1])

    sampled_tokens = payload_b
    hash_amateur = waiting_state.hash

    assert waiting_state.identity_role == Role.AMATEUR, \
        f"waiting_state identity_role should be AMATEUR, got {getattr(waiting_state.identity_role,'value',waiting_state.identity_role)}"
    
    rep = {"kind": Kind.SEND_SAMPLED_RECV_ACK.value, "hash": hash_amateur}
    rep_b = json.dumps(rep).encode("utf-8")
    if peer_mode.get(identity_exp, "REQ") == "REQ":
        sock.send_multipart([identity_exp, b"", rep_b])
    else:
        sock.send_multipart([identity_exp,       rep_b])     
    
    rep = {
        "kind": Kind.SEND_LOGPROBS_RECV_SAMPLED.value,
        "hash": hash_expert,
        "dtype": dtype,
        "shape": shape,
    }
    rep_b = json.dumps(rep).encode("utf-8")
    if waiting_state.mode == "REQ":
        sock.send_multipart([waiting_state.identity, b"", rep_b, sampled_tokens])
    else:
        sock.send_multipart([waiting_state.identity,       rep_b, sampled_tokens])
    print(f"{BIND_ADDR}        ✅[coord: commit_sampled_tokens] committed sampled tokens for hash={hash_expert}")

            
def send_logits(identity, header, payload_b,sock, waiting_state):        
    if waiting_state.identity_role == Role.EXPERT: # Expert is waiting
        identity_exp = waiting_state.identity
        dtype = header.get("dtype")
        shape = header.get("shape")
        hash_ama = header.get("hash")
        logits = payload_b
    else: # Amateur is waiting
        identity_exp = identity
        dtype = waiting_state.ama_logits.dtype
        shape = waiting_state.ama_logits.shape
        hash_ama = waiting_state.hash
        logits = waiting_state.ama_logits.logits
    
    rep_h = {
        "kind": Kind.SEND_WAITING_RECV_LOGITS.value,
        "dtype": dtype,
        "shape": shape,
        "hash": hash_ama,
    }
    rep_h_b = json.dumps(rep_h).encode("utf-8")
    payload = logits
    mode = peer_mode.get(identity_exp, "REQ")
    if mode == "REQ":
        sock.send_multipart([identity_exp, b"", rep_h_b, payload])
    else:
        sock.send_multipart([identity_exp,       rep_h_b, payload])
    print(f"{BIND_ADDR}    [coord: send_logits] sent logits, shape={shape}, dtype={dtype}")

def main():
    
    ctx = zmq.Context.instance()
    sock = ctx.socket(zmq.ROUTER)
    sock.bind(BIND_ADDR)
    print(f"{BIND_ADDR}🚀[coord] ROUTER listening on {BIND_ADDR}")
    waiting_state: WaitingState = WaitingState()

    while True:
        parts = sock.recv_multipart()
        if not parts or len(parts) < 2:
            continue
        identity, header_b, payload_b = parse_incoming(parts)
        header = json.loads(header_b.decode("utf-8") or "{}")
        kind = Kind(header.get("kind"))
        hash = header.get("hash", -1)   
        if kind == Kind.SEND_LOGPROBS_RECV_SAMPLED: #From Amateur
            if waiting_state.waiting_kind == Kind.SEND_WAITING_RECV_LOGITS: #Expert is waiting
                assert hash == waiting_state.hash, \
                    f"hash mismatch: got {hash}, waiting for {waiting_state.hash}"
                # send logits to waiting requester
                send_logits(identity,header,payload_b,sock,waiting_state)
                # update waiting state
                waiting_state = WaitingState(
                    hash=hash,
                    waiting_kind=Kind.SEND_LOGPROBS_RECV_SAMPLED,
                    identity=identity,
                    identity_role=Role.AMATEUR,
                    mode=peer_mode.get(identity, "REQ"),
                )
            else: #No one is waiting
                waiting_state = WaitingState(
                    hash=hash,
                    waiting_kind=Kind.SEND_LOGPROBS_RECV_SAMPLED,
                    identity=identity,
                    identity_role=Role.AMATEUR,
                    mode=peer_mode.get(identity, "REQ"),
                    ama_logits=Logits(logits=payload_b,
                                  dtype=header.get("dtype"),
                                  shape=header.get("shape")),
                )
        elif kind == Kind.SEND_WAITING_RECV_LOGITS: # From Expert
            if waiting_state.waiting_kind == Kind.SEND_LOGPROBS_RECV_SAMPLED:  #Amateur is waiting
                assert hash == waiting_state.hash, \
                    f"hash mismatch: got {hash}, waiting for {waiting_state.hash}"
                send_logits(identity,header,payload_b,sock,waiting_state)
            else: #No one is waiting
                waiting_state = WaitingState(
                    hash=hash,
                    waiting_kind=Kind.SEND_WAITING_RECV_LOGITS,
                    identity=identity,
                    identity_role=Role.EXPERT,
                    mode=peer_mode.get(identity, "REQ"),
                )           
        elif kind == Kind.SEND_SAMPLED_RECV_ACK: # From Expert
            # Amateur should be waiting
            assert waiting_state.waiting_kind == Kind.SEND_LOGPROBS_RECV_SAMPLED, \
                f"Amateur should be waiting in SEND_SAMPLED_RECV_ACK, but got {getattr(waiting_state.waiting_kind,'value',waiting_state.waiting_kind)}"
            assert hash == waiting_state.hash, \
                f"hash mismatch in SEND_SAMPLED_RECV_ACK: got {hash}, waiting for {waiting_state.hash}"
            commit_sampled_tokens(identity, header, payload_b,sock, waiting_state)
            # clear waiting state
            waiting_state = WaitingState()
        else:
            assert False, f"unknown kind: {kind}"

if __name__ == "__main__":
    main()
