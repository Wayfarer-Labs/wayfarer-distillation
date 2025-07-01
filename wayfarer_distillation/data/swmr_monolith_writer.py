#!/usr/bin/env python3
"""swmr_monolith_writer.py ────────────────────────────────────────────────
End‑to‑end **Self‑Forcing ODE pair** logger for WAN‑2.1‑T2V using a
*monolithic* HDF5 layout **and** true SWMR.

The design ➡ one dedicated **writer** process owns the HDF5 file; one
worker process per GPU renders samples and pushes dictionaries through a
`multiprocessing.Queue`.  This avoids the *"Cannot re‑initialize CUDA in
forked subprocess"* pitfall:  CUDA is only touched *inside* the spawned
workers, and the start method is forced to **spawn**.

Usage (8 GPU node):
```
python swmr_monolith_writer.py \
       --prompts prompts.txt \
       --ckpt-dir /path/to/wan/checkpoints \
       --out ode_pairs.h5 \
       --cfg /path/to/config.py \
       --gpus 8
```
The script auto‑detects shapes from the WAN config; edit the constants
near the top if you switch resolution buckets.
"""
# ---------------------------------------------------------------------------
from __future__ import annotations
import argparse, os, sys, textwrap, time, json, math, gc
import multiprocessing as mp
import pathlib
from typing import List, Dict, Any

import h5py, numpy as np
import torch

# ---------------------------------------------------------------------------
# WAN‑T2V imports – ensure PYTHONPATH contains the repo root or install as pkg
from wan.configs import t2v_1_3B
from wan.text2video import WanT2V
def load_config(): return t2v_1_3B
MAX_PROMPTS = 1_000
# ---------------------------------------------------------------------------
# 1.  HDF5 helpers
# ---------------------------------------------------------------------------
DT_STR   = h5py.string_dtype(encoding="utf-8")
DT_META  = np.dtype([("seed", "<i8"), ("guidance", "<f4"),
                     ("gpu", "<i2"),  ("wall_ms", "<i4")])

def _create_dsets(f: h5py.File, shapes: Dict[str, tuple[int, ...]]):
    """Create extendible, chunked datasets the first time we open the file."""
    max_n = None  # unlimited along axis 0

    # prompts – variable‑length UTF‑8
    f.create_dataset("prompts", shape=(0,), maxshape=(max_n,), dtype=DT_STR)

    # sample‑level meta packed into compound dtype
    f.create_dataset("sample_attrs", shape=(0,), maxshape=(max_n,),
                     dtype=DT_META)

    # latents + velocities – float16, LZF for speed
    f.create_dataset("latents", shape=(0,)+shapes["latents"],
                     maxshape=(max_n,)+shapes["latents"], dtype="float32",
                     chunks=(1,)+shapes["latents"], compression="gzip")

    for name in ("velocity_cond", "velocity_uncond"):
        f.create_dataset(name, shape=(0,)+shapes[name],
                         maxshape=(max_n,)+shapes[name], dtype="float32",
                         chunks=(1,)+shapes[name], compression="gzip")

    f.create_dataset("timesteps", shape=(0, shapes["timesteps"][0]),
                     maxshape=(max_n, shapes["timesteps"][0]), dtype="int32")


def _append_sample(f: h5py.File, idx: int, prompt: str, meta: np.ndarray,
                   lat: np.ndarray,
                   v_c: np.ndarray, v_u: np.ndarray,
                   steps: np.ndarray):
    """Write a single sample (no resize here – caller already resized)."""
    f["prompts"][idx]       = prompt
    f["sample_attrs"][idx]  = meta
    f["latents"][idx]       = lat
    f["velocity_cond"][idx] = v_c
    f["velocity_uncond"][idx]= v_u
    f["timesteps"][idx]     = steps

# ---------------------------------------------------------------------------
# 2.  Multiprocessing workflow
# ---------------------------------------------------------------------------

def writer_proc(out_path: str, shapes: Dict[str, tuple[int, ...]],
                queue: mp.Queue, n_workers: int):
    """Single process owning the HDF5 file (SWMR writer)."""
    with h5py.File(out_path, "a", libver="latest") as f:
        if "prompts" not in f:
            _create_dsets(f, shapes)
        f.swmr_mode = True

        finished = 0
        while finished < n_workers:
            item = queue.get()
            if item is None:  # sentinel
                finished += 1
                continue

            prompt, attrs, tensors = item
            # tensors unpack
            lat, v_c, v_u, steps = tensors
            idx = f["latents"].shape[0]
            # resize all extendible datasets once
            for ds in ("prompts", "sample_attrs", "latents",
                       "velocity_cond", "velocity_uncond", "timesteps"):
                f[ds].resize(idx+1, axis=0)
            _append_sample(f, idx, prompt, attrs, lat, v_c, v_u, steps)
            f.flush()  # make visible to readers

        print("[writer] All workers done – closing file.")


def worker_proc(rank: int, args: argparse.Namespace,
                shapes: Dict[str, tuple[int, ...]], queue: mp.Queue):
    """One GPU = one process ⇒ generate, postprocess, enqueue."""
    torch.cuda.set_device(rank)

    # ---------------------------  init WAN‑T2V  ---------------------------
    cfg = load_config()
    model = WanT2V(cfg, checkpoint_dir=args.ckpt_dir,
                   pretrained_model_name_or_path=args.model_hf_path,
                   device_id=rank,
                   rank=0,
                   t5_fsdp=False,
                   )

    # ---------------------------  prompt sharding  ------------------------
    with open(args.prompts, "r", encoding="utf-8") as fh:
        prompt_list = [ln.strip() for ln in fh if ln.strip()]
        prompt_list = prompt_list[:MAX_PROMPTS]
    shard = [p for i, p in enumerate(prompt_list) if i % args.gpus == rank]

    print(f"[GPU {rank}] {len(shard)} prompts → generate …")

    for i, prompt in enumerate(shard):
        t0 = time.perf_counter()
        sample = model.generate(prompt, size=(832, 480), frame_num=81, shift=8.0, sample_solver='unipc', sampling_steps=50, guide_scale=6.0, n_prompt="", seed=-1, offload_model=True, return_ode_distill_data=True)
        wall_ms = int((time.perf_counter() - t0) * 1000)

        # --------------  prepare numpy for HDF5  --------------
        lat = sample["denoised_latents"].float().cpu().numpy().astype(np.float32)
        v_c = np.stack([v.float().cpu().numpy().astype(np.float32)
                         for v in sample["velocity_cond"]])
        v_u = np.stack([v.float().cpu().numpy().astype(np.float32)
                         for v in sample["velocity_uncond"]])
        steps = np.array(torch.stack(sample["accum_timesteps"]).cpu(), dtype="int32")

        meta = np.array((0, 5.0, rank, wall_ms), dtype=DT_META)

        queue.put((prompt, meta, (lat, v_c, v_u, steps)))
        print(f"[GPU {rank}] {i} / {len(shard)} prompts done")
        # free GPU mem quickly
        # del sample; torch.cuda.empty_cache(); gc.collect()

    queue.put(None)  # sentinel
    print(f"[GPU {rank}] done.")

# ---------------------------------------------------------------------------
# 3.  Entry‑point
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    from wan.configs.globals import BASE_DIR
    p = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=textwrap.dedent("""
        Render WAN‑2.1 samples and store ODE distillation tuples in a
        single HDF5 file (monolithic, SWMR).  One writer process keeps
        the file open; each GPU process streams its results through an
        inter‑process queue.
        """))
    p.add_argument("--prompts", default=pathlib.Path('filtered_text_prompts_16k.txt'), help="txt file – one prompt per line")
    p.add_argument("--ckpt-dir", default=pathlib.Path(BASE_DIR), help="WAN checkpoint directory")
    p.add_argument("--model-hf-path", default="Wan-AI/Wan2.1-T2V-1.3B", help="WAN model Hugging Face path")
    p.add_argument("--out",       default=pathlib.Path('wayfarer_distillation/data/wan_ode_pairs.h5'), help="output HDF5 filename")
    p.add_argument("--gpus", type=int, default=8, help="#GPUs to use")
    return p.parse_args()


def infer_shapes(cfg) -> Dict[str, tuple[int, ...]]:
    """Derive dataset shapes from WAN config (hard‑coded fallback)."""
    F, H, W      = 81, 480, 832
    C, F_lat     = 16, 21
    H_lat, W_lat = 60, 104
    S            = 50
    return {
        "latents"       : (C, F_lat, H_lat, W_lat),
        "velocity_cond" : (S, C, F_lat, H_lat, W_lat),
        "velocity_uncond":(S, C, F_lat, H_lat, W_lat),
        "timesteps"     : (S,),
    }


def main():
    args = parse_args()
    assert not os.path.exists(args.out), f"File {args.out} already exists"
    # *Must* come before any CUDA ops when using multiprocessing
    mp.set_start_method("spawn", force=True)

    shapes = infer_shapes(None)  # TODO: derive from cfg if variable

    queue   = mp.Queue(maxsize=args.gpus * 2)
    writer  = mp.Process(target=writer_proc,
                         args=(args.out, shapes, queue, args.gpus),
                         daemon=False)
    writer.start()

    workers = []
    for rank in range(args.gpus):
        p = mp.Process(target=worker_proc,
                       args=(rank, args, shapes, queue), daemon=False)
        p.start(); workers.append(p)

    for p in workers: p.join();  writer.join()

    print("All processes finished ✅")


if __name__ == "__main__":
    main()
