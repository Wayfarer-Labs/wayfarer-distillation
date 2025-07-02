import time
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

class Timer:
    def __init__(self):
        self.prev_time = time.time()
    
    def hit(self):
        ping = (new_time := time.time()) - self.prev_time
        self.prev_time = new_time
        return ping


def barrier():
    if dist.is_initialized(): dist.barrier()

def module_from(module: torch.nn.Module | DistributedDataParallel) -> torch.nn.Module:
    if isinstance(module, DistributedDataParallel): return module.module
    return module

def setup(force=False):
    if not force:
        try:
            dist.init_process_group(backend="nccl")
            global_rank = dist.get_rank()
            local_rank = int(os.environ.get("LOCAL_RANK", 0))
            world_size = dist.get_world_size()
            
            return global_rank, local_rank, world_size
        except:
            import traceback
            traceback.print_exc()
            return 0, 0, 1
    else:
        dist.init_process_group(backend="nccl")
        global_rank = dist.get_rank()
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        world_size = dist.get_world_size()
        
        return global_rank, local_rank, world_size
        
def cleanup():
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()
