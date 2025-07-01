import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from owl_wms.models.gamerft_audio import GameRFTAudio
from owl_wms.utils import (
    freeze, unfreeze,
    SamiTimer as Timer, versatile_load
)
from torch.nn.parallel import DistributedDataParallel
from owl_wms.trainers.base import BaseTrainer
from owl_wms.configs import TrainingConfig, TransformerConfig as ModelConfig, WANDBConfig as LoggingConfig

def module_from_ddp(model: GameRFTAudio | DistributedDataParallel) -> GameRFTAudio:
    return model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model


class Loss_WAN_Distillation(nn.Module):
    def __init__(self,
            teacher:   GameRFTAudio,
            student:   GameRFTAudio,
            normalize:          bool = False,
            normalize_eps: float      = 1e-6,
            debug_basic: bool = False,
        ):
        super().__init__()
        self.teacher   = teacher ; freeze(self.teacher)
        self.student   = student

        # -- 
        self.normalize          = normalize
        self.normalize_eps      = normalize_eps
        self.debug_basic        = debug_basic

    def loss_ode_regression(self,
            teacher_clip_inputs: Tensor,   # [(b*d), n, c, h, w] where d is denoising steps. contains denoised iamges and not velocities
            teacher_clip_targets: Tensor,  # ^
            timesteps_start: Tensor,       # [(b*d), n, 1]
            timesteps_end: Tensor,         # [(b*d), n, 1]
            mouse: Tensor, # [(b*d) n 2]
            btn: Tensor,   # [(b*d) n 11]
        ) -> dict[str, Tensor]:
        assert sum(x.numel() for x in self.student.parameters() if x.requires_grad) > 0, 'student must not be frozen'
        assert sum(x.numel() for x in self.teacher.parameters() if x.requires_grad) == 0, 'teacher must be frozen'
        student = module_from_ddp(self.student)
        dt = timesteps_start - timesteps_end

        # -- cant calculate the loss directly since the kv cache was warmed up with a different batch-size that
        # does not include the denoising steps. therefore we sliding-window over the batch-dimension where window_size=batch_size
        denoising_steps = teacher_clip_inputs.shape[1]
        clip_loss  = torch.tensor(0., device=teacher_clip_inputs.device)

        for d in range(denoising_steps):
            x_t_d   = teacher_clip_inputs [:, d, ::]
            t_d     = timesteps_start     [:, d, ::]
            mouse_d = mouse               [:, d, ::]
            btn_d   = btn                 [:, d, ::]
            dt_d    = dt                  [0, d, 0].item()

            velocity_clip_student, __doc__ = student.velocity_fn(x_t = x_t_d, t = t_d,
                                                                                    mouse = mouse_d, btn = btn_d,
                                                                                    kv_cache = None, cfg_weight = 0.0)
            # trying to minimize where the student would have taken us vs where the teacher ended up
            clip_loss  += 0.5 * F.mse_loss(teacher_clip_targets [:, d, ::], x_t_d   - (dt_d * velocity_clip_student))

        return {
            'total_loss': clip_loss,
        }


# NOTE: 
# # Goal: Evaluate whether a deep network or a wide and shallow network does better when getting distilled from WAN2.1 14B.
# # Requirements:
# A. Randomly initialized UNet - this is our hierarchical experiment.
# B. Randomly initialized ViT  - this is our depth experiment.
# C. WAN2.1 14B Frozen.
# D. WAN2.1 AutoEncoder.
# E. DataLoader that loads in video data (in the form of tensors), encodes them with the autoencoder, and then passes them into both A) and B).
# Training loop:
# For each train step:
# 1. Query dataloader to get a clip of raw frames by mmap on a `.pt` tensor.
# 2. Encode the raw frames with D.
# 3. Pass the encoded frames into A and B and C to get velocities *at each denoising step*.
# 4. Calculate loss on A and C, then do a backwards step.
# 5. Calculate loss on B and C, then do a backwards step.
# 6. Repeat.


class WAN_DistillationTrainer(BaseTrainer):
    def __init__(self,
            train_cfg:      TrainingConfig,
            logging_cfg:    LoggingConfig,
            model_cfg:      ModelConfig,
            global_rank:    int = 0,
            local_rank:     int = 0,
            world_size:     int = 1
        ):
        super().__init__(train_cfg, logging_cfg, model_cfg, global_rank, local_rank, world_size)