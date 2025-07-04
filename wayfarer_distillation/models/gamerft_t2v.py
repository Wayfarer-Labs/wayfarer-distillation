import random
from contextlib import contextmanager
import sys
import torch
from tqdm import tqdm
from torch import nn
import torch.nn.functional as F
from typing import Literal
import gc
import torch.distributed as dist
import einops as eo
import math
from wayfarer_distillation.nn.embeddings import (
    TimestepEmbedding,
    LearnedPosEnc,
    TextConditioningEmbedding
)
from wayfarer_distillation.nn.attn import UViT, FinalLayer, DiT
from wan.modules.vae import WanVAE
from wan.configs import t2v_1_3B, i2v_14B, t2v_14B, i2v_14B
from wan.utils.fm_solvers import (FlowDPMSolverMultistepScheduler,
                               get_sampling_sigmas, retrieve_timesteps)
from wan.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler
from wayfarer_distillation.configs import TransformerConfig

class GameRFT_T2V(nn.Module):
    def __init__(self, cfg: TransformerConfig,
                 vae: WanVAE,
                 wan_config = t2v_1_3B, # bad design but used to distill):
                 device: torch.device = torch.device('cuda:0'),
                 rank: int = 0):
        super().__init__()
        backbone = cfg.backbone
        self.rank = rank
        self.device = device
        self.config: TransformerConfig = cfg
        self.wan_config = wan_config
        self.backbone = UViT(cfg) if backbone == "uvit" else DiT(cfg)
        self.vae = vae
        self.param_dtype = t2v_1_3B.param_dtype
        self.num_train_timesteps = t2v_1_3B.num_train_timesteps
        self.text_cond = TextConditioningEmbedding(
            in_dim=cfg.text_dim,           # e.g. 4096
            out_dim=cfg.d_model,
            num_heads=cfg.text_heads,
        )
        self.t_embed = TimestepEmbedding(out_dim=cfg.d_model)

        self.proj_in  = nn.Linear(cfg.channels, cfg.d_model, bias=False)
        self.proj_out = FinalLayer(None, cfg.d_model, cfg.channels) # TODO first argument is unused

    # ------------------------------------------------------------------
    def velocity_fn(self, latents, ts, text_tokens):
        B, N, C, H, W = latents.shape

        # ----- cond vectors ------------------------------------------------
        text_cond = self.text_cond(text_tokens, n_frames=N)  # [B, N, d]
        time_cond = self.t_embed(ts)                        # [B, N, d]
        cond = text_cond + time_cond                        # [B, N, d]

        # ----- rearrange video tokens -------------------------------------
        x = eo.rearrange(latents, "b n c h w -> b (n h w) c")     # [B, V, C]
        x = self.proj_in(x)

        # ----- backbone ----------------------------------------------------
        eps = self.backbone(x, cond)                    # [B, V, d]
        eps = self.proj_out(eps, cond)                  # [B, V, C]
        eps = eo.rearrange(eps, "b (n h w) c -> b n c h w", n=N, h=H, w=W)
        return eps

    def forward(self, latents, t, text_tokens):
        return self.velocity_fn(latents, t, text_tokens)

    def generate(self,
                starting_noise,
                text_tokens, # [B, num_tokens, D=4096]
                negative_tokens, # [B, num_tokens, D=4096],
                frame_num=81,
                shift=5.0,
                sample_solver='unipc',
                sampling_steps=50,
                guide_scale=5.0, 
                seed=-1,
                return_ode_distill_data=False,
                offload_model=True):

        F = frame_num

        seed = seed if seed >= 0 else random.randint(0, sys.maxsize)
        seed_g = torch.Generator(device=self.device)
        seed_g.manual_seed(seed)
        
        # TODO send to cuda rank
        context = text_tokens
        context_null = negative_tokens

        noise = starting_noise

        @contextmanager
        def noop_no_sync():
            yield

        no_sync = getattr(self, 'no_sync', noop_no_sync)

        with torch.amp.autocast(device_type='cuda', dtype=self.param_dtype), no_sync():
            if sample_solver == 'unipc':
                sample_scheduler = FlowUniPCMultistepScheduler(
                    num_train_timesteps=self.num_train_timesteps,
                    shift=1,
                    use_dynamic_shifting=False)
                sample_scheduler.set_timesteps(
                    sampling_steps, device=self.device, shift=shift)
                timesteps = sample_scheduler.timesteps
            elif sample_solver == 'dpm++':
                sample_scheduler = FlowDPMSolverMultistepScheduler(
                    num_train_timesteps=self.num_train_timesteps,
                    shift=1,
                    use_dynamic_shifting=False)
                sampling_sigmas = get_sampling_sigmas(sampling_steps, shift)
                timesteps, _ = retrieve_timesteps(
                    sample_scheduler,
                    device=self.device,
                    sigmas=sampling_sigmas)
            else:
                raise NotImplementedError("Unsupported solver.")

            latents = noise

            velocity_cond = []
            velocity_uncond = []
            accum_timesteps = []
            
            # -- denoising 
            for _, t in enumerate(tqdm(timesteps)):
                latent_model_input = latents
                timestep = [t]

                timestep = torch.stack(timestep)

                self.to(self.device)
                noise_pred_cond = self.forward(
                    latent_model_input, t=timestep, text_tokens=context)
                noise_pred_uncond = self.forward(
                    latent_model_input, t=timestep, text_tokens=context_null)
                if return_ode_distill_data:
                    velocity_cond.append(noise_pred_cond.cpu()) # -- offload
                    velocity_uncond.append(noise_pred_uncond.cpu())
                    accum_timesteps.append(t)

                noise_pred = noise_pred_uncond + guide_scale * (
                    noise_pred_cond - noise_pred_uncond)

                temp_x0 = sample_scheduler.step(
                    noise_pred.unsqueeze(0),
                    t,
                    latents[0].unsqueeze(0),
                    return_dict=False,
                    generator=seed_g)[0]
                latents = temp_x0.squeeze(0)

        if offload_model:
            self.cpu()


        del latents
        del sample_scheduler

        if offload_model:
            gc.collect()
            torch.cuda.synchronize()

        if dist.is_initialized():
            dist.barrier()

        if self.rank != 0:
            return None
        elif return_ode_distill_data:
            return {
                "starting_noise": noise,
                "velocity_cond": torch.stack(velocity_cond), # list[50] of [16,21,60,104]
                "velocity_uncond": torch.stack(velocity_uncond), # list[50] of [16,21,60,104]
                "accum_timesteps": accum_timesteps, # list[50] of [1]
            }
        else: return None


if __name__ == "__main__":
    from wayfarer_distillation.configs import Config

    cfg = Config.from_yaml("basic.yml").model
    model = GameRFT_T2V(cfg).cuda().bfloat16()

    with torch.no_grad():
        x = torch.randn(1, 128, 16, 256, device='cuda', dtype=torch.bfloat16)
        mouse = torch.randn(1, 128, 2, device='cuda', dtype=torch.bfloat16) 
        btn = torch.randn(1, 128, 11, device='cuda', dtype=torch.bfloat16)
        
        loss = model(x, mouse, btn)
        print(f"Loss: {loss.item()}")