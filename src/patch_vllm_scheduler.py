# This file overrides the Scheduler in vLLM v0.11.0
# ref: https://github.com/vllm-project/vllm/blob/v0.11.0/vllm/v1/core/sched/scheduler.py

# This schedule patch implements synchronization between the Expert and Amateur schedulers. In particular, it:
# - blocks scheduling until the initial batch is fully scheduled, ensuring
#   that both schedulers observe the same initial running-request state
# - passes the number of preemptible running requests from the Expert to
#   the Amateur via shared memory for coordinated preemption control


import sys
import vllm
import os
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.core.kv_cache_manager import KVCacheManager
from vllm.config import VllmConfig
from vllm.multimodal import MULTIMODAL_REGISTRY, MultiModalRegistry
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.structured_output import StructuredOutputManager
from vllm.v1.core.sched.output import SchedulerOutput

from typing import Optional

from vllm.v1.request import Request
from vllm.v1.core.kv_cache_manager import KVCacheBlocks

import mmap
from ipc import init_scheduler_ipc
import numpy as np


try:
    INITIAL_BATCH_SIZE = int(os.environ.get("VLLM_BATCH_SIZE"))
    JOB_ID = os.environ.get("JOB_ID")
except KeyError as e:
    raise RuntimeError(f"❌ Required environment variable not set: {e.args[0]}")
print(f"😀[patch_vllm_scheduler] INITIAL_BATCH_SIZE={INITIAL_BATCH_SIZE}")


def install_patch(role="Expert"):
    if role not in ("Expert", "Amateur"):
        raise ValueError("role must be 'Expert' or 'Amateur'")
    VLLM_MIN = (0, 11, 0) # vLLM minimum version required for this patch
    # check vLLM version
    ver = tuple(int(x) for x in getattr(vllm, "__version__", "0.0.0").split(".")[:3])
    if ver < VLLM_MIN:
        print(f"[patch] vLLM {ver} < min {VLLM_MIN}; patch may not match", file=sys.stderr)
        
    # Create or connect to shared memory and semaphores to synchronize Expert and Amateur schedulers
    shm, sem_ready, sem_go = init_scheduler_ipc(is_owner=(role=="Expert"), JOB_ID=JOB_ID,N=INITIAL_BATCH_SIZE)
    mm = mmap.mmap(shm.fd, shm.size)
    shm.close_fd()
    
    # patch
    orig_init = Scheduler.__init__
    orig_schedule = Scheduler.schedule        
    orig_allocate_slots = KVCacheManager.allocate_slots
            
    AMATEUR_RUNNING = 0
    EXPERT_RUNNING = 0
    
    def _init_patch(
        self,
        vllm_config: VllmConfig,
        kv_cache_config: KVCacheConfig,
        structured_output_manager: StructuredOutputManager,
        mm_registry: MultiModalRegistry = MULTIMODAL_REGISTRY,
        include_finished_set: bool = False,
        log_stats: bool = False,
    ) -> None:
        print(f"[patch] Patching Scheduler.__init__ to set initial_batch_size={INITIAL_BATCH_SIZE}")
        self.initial_batch_size = INITIAL_BATCH_SIZE
        self._warmup_done = False
        self.waiting_saved = None
        
        setup_initial_parameters = orig_init(
            self,
            vllm_config,
            kv_cache_config,
            structured_output_manager,
            mm_registry,
            include_finished_set,
            log_stats,
        )
        self.waiting_empty = self.waiting.copy()
        
        return setup_initial_parameters


    def _schedule_patch(self) -> SchedulerOutput:
        global EXPERT_RUNNING,AMATEUR_RUNNING
        if len(self.waiting) == 0 and len(self.running) == 0:
            self._warmup_done = False
        if not self._warmup_done:
            assert len(self.running) == 0, "running should be empty during warmup"
            if len(self.waiting) < self.initial_batch_size:
                self.waiting_saved = self.waiting.copy()
                self.waiting = self.waiting_empty # temporarily set waiting to empty to avoid scheduling
                assert len(self.waiting) == 0, "waiting should be empty during warmup"
            else:
                self.waiting_saved = None
                self._warmup_done = True
                print(f"[debug] Scheduler warmup done with {len(self.waiting)} requests.")
        
        
        # print(f"😀 running: {len(self.running)} waiting: {len(self.waiting)}")

        if self.waiting_saved is not None:
            scheduler_output = orig_schedule(self)
            self.waiting = self.waiting_saved 
        else:
            if role == "Amateur":
                sem_ready.release()
                # wait for Expert to write running request ids
                sem_go.acquire()
                arr = np.ndarray(
                    shape=(INITIAL_BATCH_SIZE,),
                    dtype=np.float32,
                    buffer=mm
                )
                valid_ids = [
                    str(int(x))
                    for x in arr
                    if x >= 0
                ]
     
                EXPERT_RUNNING = len(valid_ids)
                AMATEUR_RUNNING = 0
                scheduler_output = orig_schedule(self)
                assert EXPERT_RUNNING == len(self.running), \
                    f"Expert and Amateur running counts differ: Expert {EXPERT_RUNNING}, Amateur {len(self.running)}"
                assert all(str(req.request_id) in valid_ids for req in self.running), \
                    "Amateur running requests differ from Expert's" 
            else: # Expert
                sem_ready.acquire()      
                
                scheduler_output = orig_schedule(self)
                
                running_req_ids = [req.request_id for req in self.running]
                buf = np.full(INITIAL_BATCH_SIZE, -1, dtype=np.float32)
                for i, rid in enumerate(running_req_ids):
                    if i >= INITIAL_BATCH_SIZE:
                        break
                    buf[i] = float(int(rid))
                arr = np.ndarray(
                    shape=(INITIAL_BATCH_SIZE,),
                    dtype=np.float32,
                    buffer=mm
                )
                arr[:] = buf

                sem_go.release()

        return scheduler_output
      
    
    def _allocate_slots_patch( self,
        request: Request,
        num_new_tokens: int,
        num_new_computed_tokens: int = 0,
        new_computed_blocks: Optional[KVCacheBlocks] = None,
        num_lookahead_tokens: int = 0,
        delay_cache_blocks: bool = False,
        num_encoder_tokens: int = 0,
    ) -> Optional[KVCacheBlocks]:
        global AMATEUR_RUNNING, EXPERT_RUNNING
        
        if role == "Amateur":
            AMATEUR_RUNNING += 1
            if AMATEUR_RUNNING > EXPERT_RUNNING:
                # print(f"😀[patch_vllm_scheduler] Amateur preempting request {request.request_id}, Amateur Ranning{AMATEUR_RUNNING}, Expert Runnning{EXPERT_RUNNING}")
                AMATEUR_RUNNING -= 1
                return None  # Preempt the request by returning None 
        
        new_blocks = orig_allocate_slots(
            self,
            request,
            num_new_tokens,
            num_new_computed_tokens,
            new_computed_blocks,
            num_lookahead_tokens,
            delay_cache_blocks,
            num_encoder_tokens,
        )
        
        if role == "Amateur" and new_blocks is None:
            # print(f"😀[patch_vllm_scheduler] Amateur preempting request {request.request_id} after allocation, Amateur Ranning{AMATEUR_RUNNING}, Expert Runnning{EXPERT_RUNNING}")
            AMATEUR_RUNNING -= 1
        
        return new_blocks


    Scheduler.__init__ = _init_patch
    Scheduler.schedule = _schedule_patch
    KVCacheManager.allocate_slots = _allocate_slots_patch
    print(f"😀[patch] Patched Scheduler")
    
