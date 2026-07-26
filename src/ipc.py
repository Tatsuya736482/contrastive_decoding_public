import posix_ipc
import time
import numpy as np

def init_scheduler_ipc(is_owner,JOB_ID="unknown",N=1024,dtype=np.float32):
    """
    Initialize shared memory and semaphores for scheduler IPC.
    
    :param is_owner: Whether the current process creates the IPC resources or connects to existing ones.
    :param JOB_ID: Unique identifier for the job to avoid naming conflicts.(default: "unknown")
    :param N: Number of elements in the shared memory array.(default: 1024)
    :param dtype: Data type of the elements in the shared memory array.(default: np
    """
    SHM_NAME = f"/sche_shm_{JOB_ID}"
    SEM_READY = f"/sche_sem_ready_{JOB_ID}"  # Amateur → Expert
    SEM_GO    = f"/sche_sem_go_{JOB_ID}"     # Expert → Amateur
    
    
    if is_owner:
        try:
            shm = posix_ipc.SharedMemory(
                SHM_NAME,
                posix_ipc.O_CREAT | posix_ipc.O_EXCL,
                size=N * dtype().nbytes
            )
        except posix_ipc.ExistentialError:
            shm = posix_ipc.SharedMemory(SHM_NAME)

        try:
            sem_ready = posix_ipc.Semaphore(
                SEM_READY,
                posix_ipc.O_CREAT | posix_ipc.O_EXCL,
                initial_value=0
            )
        except posix_ipc.ExistentialError:
            sem_ready = posix_ipc.Semaphore(SEM_READY)

        try:
            sem_go = posix_ipc.Semaphore(
                SEM_GO,
                posix_ipc.O_CREAT | posix_ipc.O_EXCL,
                initial_value=0
            )
        except posix_ipc.ExistentialError:
            sem_go = posix_ipc.Semaphore(SEM_GO)

    else:
        while True:
            try:
                shm = posix_ipc.SharedMemory(SHM_NAME)
                sem_ready = posix_ipc.Semaphore(SEM_READY)
                sem_go = posix_ipc.Semaphore(SEM_GO)
                break
            except posix_ipc.ExistentialError:
                time.sleep(0.001)

    return shm, sem_ready, sem_go


def cleanup_scheduler_ipc(shm, sem_ready, sem_go, mm, is_owner):
    print(f"[cleanup] called, is_owner={is_owner}")
    mm.close()
    sem_ready.close()
    sem_go.close()

    if is_owner:
        try:
            shm.unlink()
            sem_ready.unlink()
            sem_go.unlink()
            print("[patch_vllm_scheduler] ✅ IPC resources cleaned up.")
        except Exception as e:
            print(f"[patch_vllm_scheduler] ⚠️ Failed to clean up IPC resources. Error: {e}")
            pass
