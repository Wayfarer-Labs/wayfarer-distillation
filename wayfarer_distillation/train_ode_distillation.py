import os
from copy import deepcopy
import gc
from torch.nn import Linear
import wandb
import torch
from typing import TypedDict
from torch.nn.parallel import DistributedDataParallel
import torch.nn.functional as F
import h5py
import shutil
import subprocess
import time
import einops as eo
from wayfarer_distillation.train_utils  import Timer, barrier
from wayfarer_distillation.ode_pair_dataset import make_loader, ODEPairDataset
from wayfarer_distillation.models.gamerft_t2v import GameRFT_T2V
from wan.configs import t2v_1_3B, i2v_14B, t2v_14B, i2v_14B
from wan.configs.globals import BASE_DIR
from wan.modules.t5 import T5EncoderModel
from wan.modules.vae import WanVAE
from wayfarer_distillation.configs import Config
from wayfarer_distillation.train_utils import setup, cleanup

def prefix(prefix: str, info: dict[str, float]) -> dict[str, float]:
    return {f'{prefix}{k}': v for k, v in info.items()}

class ODE_Pair_Batch(TypedDict):
    starting_noise: torch.Tensor
    latents: torch.Tensor
    velocity_cond: torch.Tensor
    velocity_uncond: torch.Tensor
    timesteps: torch.Tensor
    prompts: list[str]
    prompts_enc: list[torch.Tensor]
    prompts_enc_negative: list[torch.Tensor]

class Loss_ODE_Distillation(torch.nn.Module):
    def __init__(self, apply_cfg: bool = True):
        super().__init__()
        self.apply_cfg = apply_cfg
    
    def forward(self,
                student_velocity_cond: torch.Tensor,
                student_velocity_uncond: torch.Tensor,
                teacher_velocity_cond: torch.Tensor,
                teacher_velocity_uncond: torch.Tensor) -> torch.Tensor:
        return self._loss_ode(student_velocity_cond, student_velocity_uncond, teacher_velocity_cond, teacher_velocity_uncond)

    def _loss_ode(self,
                student_velocity_cond: torch.Tensor, # (b d) n c h w 
                student_velocity_uncond: torch.Tensor, # (b d) n c h w
                teacher_velocity_cond: torch.Tensor,
                teacher_velocity_uncond: torch.Tensor) -> torch.Tensor:
        loss_cond = F.mse_loss(student_velocity_cond,
                               teacher_velocity_cond, reduction="mean")
        loss_un   = F.mse_loss(student_velocity_uncond,
                               teacher_velocity_uncond, reduction="mean")

        return 0.5 * (loss_cond + loss_un)


TRAIN_CONFIG = {
    'batch_size': 1,
    'target_batch_size': 8,
    'cfg_scale': 5,
    'max_grad_norm': 1.0,
    'log_interval': 10,
    'save_interval': 1000,
    'distill_steps': 100000,
    'lr': 1e-4,
}

class ODE_Distillation_Trainer:
    def __init__(self,
                model_uvit: GameRFT_T2V,
                model_dit: GameRFT_T2V,
                global_rank: int = 0,
                local_rank:  int = 0,
                world_size:  int = 1,
                hdf5_path: str = '/home/sky/wayfarer-distillation/wayfarer_distillation/data/new_wan_ode_pairs_1-3B.h5',
                t5_cpu: bool = True,
                offload_model: bool = True):

        # -- just cause i use a dbeugger and stopping would break things
        if global_rank == 0: run_h5clear(hdf5_path) ; barrier()

        self.train_config = TRAIN_CONFIG
        self.model_config_vit = model_uvit.config
        self.model_config_dit = model_dit.config
        self.timer = Timer()    
    
        self.rank = global_rank
        self.local_rank = local_rank
        self.world_size = world_size
        self.t5_cpu = t5_cpu
        self.offload_model = offload_model
        
        self.device     = f"cuda:{self.rank}"
        self.device     = torch.device(self.device)
        self.wan_config = t2v_1_3B
        self.text_encoder = T5EncoderModel(
            text_len=self.wan_config.text_len,
            dtype=self.wan_config.t5_dtype,
            device=torch.device('cpu') if t5_cpu else self.device,
            checkpoint_path=os.path.join(BASE_DIR, self.wan_config.t5_checkpoint),
            tokenizer_path=os.path.join(BASE_DIR, self.wan_config.t5_tokenizer),
            shard_fn=None)

        for p in self.text_encoder.model.parameters():
            p.requires_grad = False

        self.student_model_uvit = model_uvit.to(self.device)
        self.student_model_dit  = model_dit.to(self.device)

        if world_size > 1:
            self.student_model_uvit = DistributedDataParallel(self.student_model_uvit, device_ids=[self.rank])
            self.student_model_dit  = DistributedDataParallel(self.student_model_dit,  device_ids=[self.rank])

        # -- 
        self.train_steps    = 0
        self.max_steps      = self.train_config['distill_steps']
        self.log_interval   = self.train_config['log_interval']
        self.save_interval  = self.train_config['save_interval']

        self.opt_vit = torch.optim.AdamW(self.student_model_uvit.parameters(), lr=self.train_config['lr'])
        self.opt_dit = torch.optim.AdamW(self.student_model_dit.parameters(), lr=self.train_config['lr'])
        self.scaler_vit = torch.amp.GradScaler()
        self.scaler_dit = torch.amp.GradScaler()
        self.ctx = torch.amp.autocast(device_type=self.device.type, dtype=torch.bfloat16)

        self.batch_size = self.train_config['batch_size']
        self.target_batch_size = self.train_config['target_batch_size']
        self.cfg_scale = None # initialize from batch  
        self.max_grad_norm = self.train_config['max_grad_norm']

        self._loader = make_loader(hdf5_path, self.batch_size, num_workers=4, pin=True)
        self.dataset: ODEPairDataset = self._loader.dataset
        self.iter_loader = iter(self._loader)
        self.loss_fn: Loss_ODE_Distillation = Loss_ODE_Distillation()

    @property
    def save_path(self):
        return f"{self.train_config['checkpoint_dir']}/ode_distill_step_{self.train_steps}.pt"

    @property
    def should_train(self): return self.train_steps < self.max_steps

    @property
    def should_save(self):  return self.train_steps % self.save_interval == 0

    @property
    def should_log(self):   return self.train_steps % self.log_interval == 0
    

    def _encode_prompts(self, prompts: list[str]) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        if not self.t5_cpu:
            self.text_encoder.model.to(self.device)
            prompts_enc          = self.text_encoder(prompts, self.device) 
            prompts_enc_negative = self.text_encoder([""] * len(prompts), self.device)
            if self.offload_model: self.text_encoder.model.cpu()
        else:
            prompts_enc          = self.text_encoder(prompts, torch.device('cpu'))
            prompts_enc_negative = self.text_encoder([""] * len(prompts), torch.device('cpu'))
            prompts_enc          = [t.to(self.device) for t in prompts_enc]
            prompts_enc_negative = [t.to(self.device) for t in prompts_enc_negative]

        return prompts_enc, prompts_enc_negative


    def _format_batch(self) -> ODE_Pair_Batch:
        try: batch = next(self.iter_loader)
        except StopIteration: self.iter_loader = iter(self._loader) ; return self._format_batch()
        
        if self.cfg_scale is None: # NOTE Big assumption is that they all have the same guidance
            self.cfg_scale = batch['attrs']['guidance'][0].item()

        prompts_enc, prompts_enc_negative = None, None

        if 'prompts_enc' in batch and 'prompts_enc_negative' in batch:
            prompts_enc          = batch['prompts_enc']
            prompts_enc_negative = batch['prompts_enc_negative']
        else:
            prompts_enc, prompts_enc_negative = self._encode_prompts(batch['prompts'])
            self.dataset.writeback_prompts(batch['index'], prompts_enc, prompts_enc_negative)

        for k in ('starting_noise', 'latents'):
            batch[k] = eo.rearrange(batch[k].to(self.device), 'b c n (h p1) (w p2) -> b n (p1 p2 c) h w', p1=4, p2=4) # deviating from wan for memory saving without sequence parallel

        for k in ('velocity_cond', 'velocity_uncond'):
            batch[k] = eo.rearrange(batch[k].to(self.device), 'b d c n (h p1) (w p2) -> b d n (p1 p2 c) h w', p1=4, p2=4)

        return ODE_Pair_Batch(
            starting_noise=batch['starting_noise'],
            latents=batch['latents'].to(self.device),
            velocity_cond=batch['velocity_cond'].to('cpu'),
            velocity_uncond=batch['velocity_uncond'].to('cpu'),
            timesteps=batch['timesteps'].to(self.device),
            prompts=batch['prompts'],
            prompts_enc=self.dataset._ensure_padded(prompts_enc).to(self.device),
            prompts_enc_negative=self.dataset._ensure_padded(prompts_enc_negative).to(self.device))
    

    def _optim_step(self, opt: torch.optim.Optimizer, scaler: torch.amp.GradScaler, loss: torch.Tensor) -> None:
        opt.zero_grad()
        scaler.scale(loss).backward()
        self.scaler_vit.unscale_(self.opt_vit)
        torch.nn.utils.clip_grad_norm_(self.student_model_uvit.parameters(), self.max_grad_norm)
        scaler.step(opt)
        scaler.update()
        torch.cuda.empty_cache()          # releases cached arenas
        gc.collect()                      # Python ref-counts


    def _train_step(self, model: GameRFT_T2V, opt: torch.optim.Optimizer, scaler: torch.amp.GradScaler) -> dict[str, float]:
        batch: ODE_Pair_Batch = self._format_batch()
        fwd_info = model.generate(
            starting_noise=batch['starting_noise'],
            text_tokens=batch['prompts_enc'],
            negative_tokens=batch['prompts_enc_negative'],
            guide_scale=self.cfg_scale,
            return_ode_distill_data=True,
            offload_model=True
        )

        # TODO get velocities from student
        loss = self.loss_fn.forward( # -- velcoties were offloaded so need to reload to device. 74490 before
            student_velocity_cond=eo.rearrange(fwd_info['velocity_cond'].to(self.device), 'd b n c h w -> (b d) n c h w'),
            student_velocity_uncond=eo.rearrange(fwd_info['velocity_uncond'].to(self.device), 'd b n c h w -> (b d) n c h w'),
            teacher_velocity_cond=eo.rearrange(batch['velocity_cond'].to(self.device), 'b d n c h w -> (b d) n c h w'),
            teacher_velocity_uncond=eo.rearrange(batch['velocity_uncond'].to(self.device), 'b d n c h w -> (b d) n c h w')
        )

        self._optim_step(opt, scaler, loss)
        return {'loss': loss.item(), 'time': self.timer.hit()}

    def _save_checkpoint(self) -> None:
        save_dict = {
            'model_vit': self.student_model_uvit.state_dict(),
            'model_dit': self.student_model_dit.state_dict(),
            'opt_vit': self.opt_vit.state_dict(),
            'opt_dit': self.opt_dit.state_dict(),
            'scaler_vit': self.scaler_vit.state_dict(),
            'scaler_dit': self.scaler_dit.state_dict(),
            'train_steps': self.train_steps,
            'train_config': self.train_config,
            'model_config_vit': self.model_config_vit,
            'model_config_dit': self.model_config_dit
        }

        torch.save(save_dict, self.save_path)

    def load_checkpoint(self, path: str) -> None:
        save_dict = torch.load(path, map_location=self.device)
        self.student_model_uvit.load_state_dict(save_dict['model_vit'])
        self.student_model_dit.load_state_dict(save_dict['model_dit'])
        self.opt_vit.load_state_dict(save_dict['opt_vit'])
        self.opt_dit.load_state_dict(save_dict['opt_dit'])
        self.scaler_vit.load_state_dict(save_dict['scaler_vit'])
        self.scaler_dit.load_state_dict(save_dict['scaler_dit'])
        self.train_steps = save_dict['train_steps']
        self.train_config = save_dict['train_config']
        self.model_config_vit = save_dict['model_config_vit']
        self.model_config_dit = save_dict['model_config_dit']

    def train(self) -> None:
        
        timer = Timer() ; s = timer.hit()
        with self.ctx:
            while self.should_train:
                # half peak activations with this ? 
                if self.train_steps % 2 == 0: info_vit = self._train_step(self.student_model_uvit, self.opt_vit, self.scaler_vit)
                else:                         info_dit = self._train_step(self.student_model_dit,  self.opt_dit, self.scaler_dit)

                if self.should_save: self._save_checkpoint() ; torch.cuda.empty_cache() ; e = timer.hit()

                barrier()

                if self.should_log:
                    wandb.log(prefix('vit_', info_vit) | prefix('dit_', info_dit))
                
                self.train_steps += 1
        
            self._save_checkpoint()
            print(f"Training complete in {self.train_steps} steps")


def run_h5clear(path):
    exe = shutil.which('h5clear')
    if exe is None:
        raise RuntimeError("h5clear not found; install hdf5-tools or use "
                           "the Python fallback.")
    subprocess.run([exe, '-s', path], check=True)


if __name__ == "__main__":
    from wan.configs.globals import BASE_DIR
    global_rank, local_rank, world_size = setup()
    device = f"cuda:{local_rank}"
    vae = WanVAE(
            vae_pth=os.path.join(BASE_DIR, t2v_1_3B.vae_checkpoint),
            device=torch.device(device))
    model_config_base = Config.from_yaml('basic.yml')
    model_config_uvit = deepcopy(model_config_base)
    model_config_dit  = deepcopy(model_config_base)
    model_config_uvit.model.backbone = 'uvit'
    model_config_dit.model.backbone = 'dit'
    uvit = GameRFT_T2V(model_config_uvit.model, vae, wan_config=t2v_1_3B, device=device)
    dit = GameRFT_T2V(model_config_dit.model, vae, wan_config=t2v_1_3B, device=device)
    trainer = ODE_Distillation_Trainer(uvit, dit, global_rank, local_rank, world_size)
    trainer.train()