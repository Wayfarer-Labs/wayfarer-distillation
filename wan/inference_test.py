import os
from configs.globals import BASE_DIR
from wan.text2video import WanT2V
from wan import configs as wan_configs
import torch
import imageio
import numpy as np

MODEL_NAME = "Wan-AI/Wan2.1-T2V-1.3B"
CHECKPOINT_DIR = BASE_DIR

model = WanT2V(
    config=wan_configs.t2v_1_3B,
    checkpoint_dir=CHECKPOINT_DIR,
    pretrained_model_name_or_path=MODEL_NAME,
    device_id=0,
    rank=0,
    t5_fsdp=False,
)

frames = model.generate(
    input_prompt="Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage.",
    size=(832, 480),
    frame_num=81,
    shift=8.0,
    sample_solver='unipc',
    sampling_steps=50,
    guide_scale=6.0,
    n_prompt="",
    seed=-1,
    offload_model=True,
    return_ode_distill_data=True,
)
# frames are c,n,480,832, C is 3, N is 81
# {'noise': torch.Size([16, 21, 60, 104]), 'velocity_cond': torch.Size([16, 21, 60, 104]), 'velocity_uncond': torch.Size([16, 21, 60, 104]), 'accum_timesteps': list of torch.Size([]), len 50, 'denoised_latents': torch.Size([16, 21, 60, 104])}

def render_video(frames_cnhw: torch.Tensor, path: str):
    import imageio
    # each image must be [NxMx3]
    frames_nchw = frames_cnhw.permute(1,2,3,0).contiguous().cpu().numpy()
    frames_nchw = (frames_nchw + 1) * 127.5

    imageio.mimsave(path, list(frames_nchw), fps=24)
# python generate.py  --task t2v-1.3B --size 832*480 --ckpt_dir ./Wan2.1-T2V-1.3B --offload_model True --t5_cpu --sample_shift 8 --sample_guide_scale 6 --prompt "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage."
