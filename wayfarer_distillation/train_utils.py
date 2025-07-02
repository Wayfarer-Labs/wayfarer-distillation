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