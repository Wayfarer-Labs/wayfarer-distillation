# hdf5_dataset_static.py
import h5py, torch
from torch.utils.data import Dataset, DataLoader
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
import numpy as np

class ODEPairDataset(Dataset):
    """
    Read-only Dataset for a *finished* monolithic HDF5 file that will NOT grow.
    """
    def __init__(self,
                 path: str,
                 text_len: int = 512,
                 hid: int = 4096,
                 dtype=torch.float16,
                 write_rank: int = 0):
        super().__init__()
        self.path       = path
        self.dtype      = dtype
        self.text_len   = text_len   # T
        self.hid        = hid        # H
        self.write_rank = write_rank
        self._h5: h5py.File | None = None   # lazy handle

    def _open(self, write: bool = False):
        """Open the file lazily; upgrade to read‑write on demand."""
        if self._h5 is None:
            mode = 'r+' if write else 'r'
            self._h5 = h5py.File(self.path, mode, libver='latest', locking=False)
            # cache handles
            self.latents : h5py.Dataset = self._h5['latents']
            self.v_cond  : h5py.Dataset = self._h5['velocity_cond']
            self.v_uncond: h5py.Dataset = self._h5['velocity_uncond']
            self.steps   : h5py.Dataset = self._h5['timesteps']
            self.prompts : h5py.Dataset = self._h5['prompts']
            self.attrs   : h5py.Dataset = self._h5['sample_attrs']
            self.noise   : h5py.Dataset = self._h5['noise']
            # optional datasets (may not exist yet)
            self._prompts_enc: h5py.Dataset  = self._h5.get('prompts_enc')
            self._prompts_neg: h5py.Dataset  = self._h5.get('prompts_enc_negative')
        elif write and self._h5.mode == 'r':
            # need a RW handle – reopen
            path = self.path
            self._h5.close(); self._h5 = None
            self._open(write=True)

        self._h5: h5py.File


    # --- Dataset API ---------------------------------------------------------
    def __len__(self):
        self._open()
        return self.latents.shape[0]

    def __getitem__(self, idx: int):
        self._open()
        idx = int(idx)

        lat = torch.from_numpy(self.latents[idx]).to(self.dtype)
        vc  = torch.from_numpy(self.v_cond[idx]).to(self.dtype)
        vu  = torch.from_numpy(self.v_uncond[idx]).to(self.dtype)
        ts  = torch.from_numpy(self.steps[idx]).to(torch.int32)
        noise = torch.from_numpy(self.noise[idx]).to(self.dtype)
        prompt_txt = self.prompts[idx].decode('utf-8')

        sample = {
            'index'           : idx,
            'starting_noise'  : noise,
            'latents'         : lat,
            'velocity_cond'   : vc,
            'velocity_uncond' : vu,
            'timesteps'       : ts,
            'prompts'         : prompt_txt,
            'attrs'           : dict(zip(self.attrs.dtype.names, self.attrs[idx]))
        }

        # attach pre‑encoded prompts if present
        if self._prompts_enc is not None:
            sample['prompts_enc']          = torch.from_numpy(self._prompts_enc[idx]).to(self.dtype)
            sample['prompts_enc_negative'] = torch.from_numpy(self._prompts_neg[idx]).to(self.dtype)
        return sample


    def __del__(self):
        if self._h5 is not None:
            self._h5.close()
    
    def writeback_prompts(self,
                        indices: torch.Tensor,            # [B] global row-ids
                        enc_pos: list[torch.Tensor] | torch.Tensor,
                        enc_neg: list[torch.Tensor] | torch.Tensor):
        """
        enc_pos / enc_neg may be
        • list of length-T_i tensors  (coming directly from WAN-T5 encode)  OR
        • already-padded [B, text_len, hid] tensor
        This helper pads (or verifies) and then performs the gather→single-writer
        write-through.
        """
        # 1. -------- ensure [B, T_MAX, H] on *CPU* -------------------------
        pos_padded = self._ensure_padded(enc_pos)
        neg_padded = self._ensure_padded(enc_neg)

        rows_cpu = indices.detach().cpu()
        pos_cpu  = pos_padded.cpu().half().contiguous()
        neg_cpu  = neg_padded.cpu().half().contiguous()

        payload = (rows_cpu, pos_cpu, neg_cpu)

        # 2. -------- single-GPU case --------------------------------------
        if not (dist.is_available() and dist.is_initialized()):
            return self._write_rows(rows_cpu.numpy(), pos_cpu.numpy(), neg_cpu.numpy())

        # 3. -------- gather to write_rank ---------------------------------
        gathered = [None] * dist.get_world_size()
        dist.gather_object(payload, gathered, dst=self.write_rank)

        if dist.get_rank() != self.write_rank:
            return                                            # non-writer done

        for pkg in gathered:
            if pkg is None: continue
            rows, pos, neg = pkg
            self._write_rows(rows.numpy(), pos.numpy(), neg.numpy())

        if self._h5 is not None:
            self._h5.flush()
        
    # ----------------------------------------------------------------------
    def _ensure_padded(self,
                    x: list[torch.Tensor] | torch.Tensor) -> torch.Tensor:
        """
        Convert *x* to a torch.float16 tensor [B, text_len, hid] on CPU.
        """
        if isinstance(x, torch.Tensor):
            # already a batch → just move to CPU/half if needed
            return x.detach()

        # x is a list of variable-length tensors
        B        = len(x)
        T_MAX    = self.text_len
        H        = x[0].shape[-1]

        out = torch.zeros((B, T_MAX, H), dtype=torch.float16)
        for i, t in enumerate(x):
            L = min(t.shape[0], T_MAX)
            out[i, :L] = t[:L]

        return out

    def _write_rows(self, rows: np.ndarray,
                    pos: np.ndarray,
                    neg: np.ndarray):
                    
        self._open(write=True)

        # create datasets lazily
        if 'prompts_enc' not in self._h5:
            N, T, H = self.__len__(), self.text_len, self.hid
            chunks  = (1, T, H)
            for name in ('prompts_enc', 'prompts_enc_negative'):
                self._h5.create_dataset(
                    name, shape=(N, T, H), dtype='float16',
                    chunks=chunks, compression='gzip', compression_opts=4)
            # update handles for __getitem__
            self._prompts_enc = self._h5['prompts_enc']
            self._prompts_neg = self._h5['prompts_enc_negative']
            
        for i, r in enumerate(rows):
            self._prompts_enc[r, :, :] = pos[i]
            self._prompts_neg[r, :, :] = neg[i]



def make_loader(path: str = '/mnt/data/ode_distillation_dataset/wan_ode_pairs.h5',
                batch_size: int = 1,
                num_workers: int = 4,
                pin: bool = True):

    dataset = ODEPairDataset(path)

    # if you launched with torch.distributed.run / torchrun
    if dist.is_available() and dist.is_initialized():
        sampler = DistributedSampler(dataset,
                                     num_replicas=dist.get_world_size(),
                                     rank=dist.get_rank(),
                                     shuffle=True,
                                     drop_last=True)
    else:
        sampler = None

    return DataLoader(dataset,
                      batch_size=batch_size,
                      sampler=sampler,
                      shuffle=(sampler is None),
                      num_workers=num_workers,
                      pin_memory=pin,
                      persistent_workers=True)
