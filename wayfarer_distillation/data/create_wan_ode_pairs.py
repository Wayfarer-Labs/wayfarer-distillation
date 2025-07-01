
import h5py
import torch
import numpy as np
from dataclasses import dataclass
from typing import Any


# dataset.h5
# ├── ode_pairs/
# │   ├── noise_pred_cond      # (16000, L, F, C, H, W) 
# │   ├── noise_pred_uncond    # (16000, L, F, C, H, W) 
# │   ├── clean_latents        # (16000, L, F, C, H, W)
# │   ├── timesteps            # (16000, L) # t in [0,1] per chunk
# │   ├── text_prompts         # (16000,) variable-length strings
# │   ├── image_prompts        # (16000, C, H, W) or None for text-only
# │   ├── has_text             # (16000,) bool tensor 
# │   ├── has_image            # (16000,) bool tensor 
# ├── metadata/
# │   ├── vae_scale_factor
# │   ├── inference_timesteps: [1000, 750, 500, 250]
# │   ├── time_shift_k: 5
# │   ├── latent_spatial_dims: [104, 60]
# │   ├── frames_per_chunk: 5 # Equal to L
# │   ├── temporal_compression: 4
# │   ├── guide_scale: 5.0
# │   └── spatial_compression: 8
# -------

def add_to_file(f: h5py.File, key: str, data: Any, override: bool = False):
    data_repr = data.shape if type(data) in [torch.Tensor, np.ndarray] else data
    if key in f:
        if not override:
            print(f"Key {key} already exists in file. Skipping...")
            return
        if override:
            print(f"Key {key} already exists in file. Overwriting {f[key]} -> {data_repr}...")
            del f[key]  # Delete first, then recreate
            
    print(f"Creating key {key} with data {data_repr}...")
    if isinstance(data, (list, tuple)) and len(data) > 0 and isinstance(data[0], str):
        # Handle variable-length strings
        dt = h5py.string_dtype(encoding='utf-8')
        f.create_dataset(key, data=data, dtype=dt)
    else:
        f.create_dataset(key, data=data, compression='gzip', chunks=True)

@dataclass
class ODE_Pair_H5PY_Config:
    destination_path: str
    num_samples: int = 16_000
    num_chunks_per_sample: int = 4  # L - fixed for simplicity
    frames_per_chunk: int = 5       # F - architecture parameter
    channels: int = 16              # C - latent channels
    image_size: tuple[int, int] = (224, 224)
    vae_scale_factor: float = 1.0
    inference_timesteps: list[int] = (1000, 750, 500, 250)
    time_shift_k: int = 5
    latent_spatial_dims: tuple[int, int] = (104, 60)
    temporal_compression: int = 4
    spatial_compression: int = 8
    guide_scale: float = 5.0

class ODE_Pair_H5PY_Dataset:
    def __init__(self, config: ODE_Pair_H5PY_Config):
        self.h5py_path = config.destination_path
        self.config = config
        
    def create_h5py_file(self, batch_size: int = 100):
        """Create HDF5 file without loading everything into memory"""
        N = self.config.num_samples
        L = self.config.num_chunks_per_sample  
        F = self.config.frames_per_chunk
        C = self.config.channels
        H, W = self.config.latent_spatial_dims
        im_H, im_W = self.config.image_size
        
        with h5py.File(self.h5py_path, 'w') as f:
            # Create metadata
            self._create_metadata(f)
            
            # Create datasets with correct shapes but don't populate yet
            f.create_dataset('ode_pairs/noise_pred_cond', 
                           shape=(N, L, F, C, H, W), 
                           dtype=np.float32,
                           compression='gzip', 
                           chunks=(min(batch_size, N), L, F, C, H, W))

            f.create_dataset('ode_pairs/noise_pred_uncond', 
                           shape=(N, L, F, C, H, W), 
                           dtype=np.float32,
                           compression='gzip', 
                           chunks=(min(batch_size, N), L, F, C, H, W))

            f.create_dataset('ode_pairs/clean_latents', 
                           shape=(N, L, F, C, H, W), 
                           dtype=np.float32,
                           compression='gzip',
                           chunks=(min(batch_size, N), L, F, C, H, W))
                           
            f.create_dataset('ode_pairs/timesteps', 
                           shape=(N, L), 
                           dtype=np.float32,
                           compression='gzip',
                           chunks=(min(batch_size, N), L))
            
            # Variable-length strings for prompts
            dt = h5py.string_dtype(encoding='utf-8')
            f.create_dataset('ode_pairs/text_prompts', 
                           shape=(N,), 
                           dtype=dt,
                           chunks=(min(batch_size, N),))
            
            f.create_dataset('ode_pairs/image_prompts', 
                           shape=(N, C, im_H, im_W), 
                           dtype=np.float32,
                           compression='gzip',
                           chunks=(min(batch_size, N), C, im_H, im_W))
                           
            f.create_dataset('ode_pairs/has_text', 
                           shape=(N,), 
                           dtype=bool,
                           chunks=(min(batch_size, N),))
                           
            f.create_dataset('ode_pairs/has_image', 
                           shape=(N,), 
                           dtype=bool,
                           chunks=(min(batch_size, N),))
        
        print(f"Created HDF5 file at {self.h5py_path} with shape:")
        print(f"  noise_pred_cond: ({N}, {L}, {F}, {C}, {H}, {W})")
        print(f"  noise_pred_uncond: ({N}, {L}, {F}, {C}, {H}, {W})")
        print(f"  clean_latents: ({N}, {L}, {F}, {C}, {H}, {W})")
        print(f"  timesteps: ({N}, {L})")
        print(f"  Ready for batch population with batch_size={batch_size}")

    def _create_metadata(self, f: h5py.File):
        """Create metadata group"""
        add_to_file(f, 'metadata/vae_scale_factor', self.config.vae_scale_factor)
        add_to_file(f, 'metadata/inference_timesteps', self.config.inference_timesteps)
        add_to_file(f, 'metadata/time_shift_k', self.config.time_shift_k)
        add_to_file(f, 'metadata/latent_spatial_dims', self.config.latent_spatial_dims)
        add_to_file(f, 'metadata/frames_per_chunk', self.config.frames_per_chunk)
        add_to_file(f, 'metadata/num_chunks_per_sample', self.config.num_chunks_per_sample)
        add_to_file(f, 'metadata/temporal_compression', self.config.temporal_compression)
        add_to_file(f, 'metadata/spatial_compression', self.config.spatial_compression)
        add_to_file(f, 'metadata/guide_scale', self.config.guide_scale)

    def add_batch(self, start_idx: int, 
                  noise_pred_cond: torch.Tensor,
                  noise_pred_uncond: torch.Tensor,
                  clean_latents: torch.Tensor, 
                  timesteps: torch.Tensor,
                  text_prompts: list[str],
                  image_prompts: torch.Tensor = None,
                  has_text: list[bool] = None,
                  has_image: list[bool] = None):
        """Add a batch of ODE pairs to the HDF5 file"""
        batch_size = noise_pred_cond.shape[0]
        end_idx = start_idx + batch_size
        
        with h5py.File(self.h5py_path, 'a') as f:  # 'a' for append mode
            f['ode_pairs/noise_pred_cond'][start_idx:end_idx] = noise_pred_cond.numpy()
            f['ode_pairs/noise_pred_uncond'][start_idx:end_idx] = noise_pred_uncond.numpy()
            f['ode_pairs/clean_latents'][start_idx:end_idx] = clean_latents.numpy()
            f['ode_pairs/timesteps'][start_idx:end_idx] = timesteps.numpy()
            f['ode_pairs/text_prompts'][start_idx:end_idx] = text_prompts
            
            if image_prompts is not None:
                f['ode_pairs/image_prompts'][start_idx:end_idx] = image_prompts.numpy()
            if has_text is not None:
                f['ode_pairs/has_text'][start_idx:end_idx] = has_text
            if has_image is not None:
                f['ode_pairs/has_image'][start_idx:end_idx] = has_image
                
        print(f"Added batch {start_idx}:{end_idx} to HDF5 file")

# Usage example:
if __name__ == "__main__":
    config = ODE_Pair_H5PY_Config(
        destination_path="ode_pairs_16k.h5",
        num_samples=16000,
        num_chunks_per_sample=4,  # Fixed 4 chunks per ODE pair
        frames_per_chunk=5        # CausVid architecture
    )
    
    dataset = ODE_Pair_H5PY_Dataset(config)
    dataset.create_h5py_file(batch_size=100)
    
    # Later, populate in batches:
    # for i in range(0, 16000, 100):
    #     batch_data = generate_ode_batch(100)  # Your ODE generation function
    #     dataset.add_batch(i, **batch_data)