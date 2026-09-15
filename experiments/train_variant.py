"""Reduced-LAM training with the VQ variants. Usage: train_variant.py NAME EPOCHS STRIDE NORMALIZE(0/1)"""
# --- portable paths / device (added when moved from the scratchpad into the repo) ---
import os as _os, sys as _sys
from pathlib import Path as _Path
import torch as _torch
_ROOT = _Path(__file__).resolve().parents[1]                     # repo root
_EXP = _ROOT / "data" / "experiments"; _EXP.mkdir(parents=True, exist_ok=True)
_sys.path.insert(0, str(_ROOT)); _sys.path.insert(0, str(_Path(__file__).parent))
_DEV = "cuda" if _torch.cuda.is_available() else ("mps" if _torch.backends.mps.is_available() else "cpu")
_WORKERS = int(_os.environ.get("NUM_WORKERS", "8" if _DEV == "cuda" else "0"))
# -----------------------------------------------------------------------------------

import sys, time; pass
S = str(_EXP)
sys.path.insert(0, S)
from dataclasses import replace
import torch
from torch.utils.data import DataLoader
from config import LAM_TRAINING, build_optimizer
from lam import LAM
from vq_variants import VQ, StrideClipDataset
from arch_variants import apply_arch
from torch.utils.tensorboard import SummaryWriter

name, epochs, stride, normalize = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), bool(int(sys.argv[4]))
entropy_weight = float(sys.argv[5]) if len(sys.argv) > 5 else 0.0
arch = sys.argv[6] if len(sys.argv) > 6 else "base"
torch.manual_seed(0)
device = torch.device(_DEV)
ds = StrideClipDataset(str(_ROOT / "data/train"), stride=stride, start_step=8 if stride > 1 else 16)
loader = DataLoader(ds, batch_size=16, shuffle=True, drop_last=False, num_workers=_WORKERS, generator=torch.Generator().manual_seed(0))
model = LAM(d_width=256, st_blocks=4, st_heads=4, head_dim=64)
model = apply_arch(model, arch)
model.cookbook = VQ(K=model.K, code_width=32, normalize=normalize, entropy_weight=entropy_weight)
model = model.to(device)
cfg = replace(LAM_TRAINING, max_lr=3e-4, min_lr=3e-4, warmup_steps=100, steps=epochs * len(loader), global_batch_size=16)
opt, sched = build_optimizer(model, cfg)
print(f"{name}: stride={stride} normalize={normalize} entropy_weight={entropy_weight} arch={arch} windows={len(ds)} batches/epoch={len(loader)}", flush=True)
history, step, t0 = [], 0, time.time()
writer = SummaryWriter(f"{S}/tb/{name}")
for epoch in range(1, epochs + 1):
    model.train(); tot = dict(loss=0., reconstruction_loss=0., vq_loss=0., perplexity=0.); n = 0
    for b, (video, _) in enumerate(loader, 1):
        video = video.to(device)
        opt.zero_grad(set_to_none=True)
        r = model(video, return_details=True)
        r["loss"].backward(); opt.step(); sched.step(); step += 1
        n += video.shape[0]
        for tag, key in (("loss/total", "loss"), ("loss/reconstruction", "reconstruction_loss"), ("loss/vq", "vq_loss"), ("codebook/perplexity_batch", "perplexity")):
            writer.add_scalar(tag, r[key].item(), step)
        for k in tot: tot[k] += r[k].detach().item() * video.shape[0]
        if b == 1 or b % 25 == 0 or b == len(loader):
            print(f"epoch {epoch}/{epochs} batch {b}/{len(loader)} loss={r['loss'].item():.5f} recon={r['reconstruction_loss'].item():.5f} vq={r['vq_loss'].item():.5f} ppl={r['perplexity'].item():.2f}", flush=True)
    history.append({k: v / n for k, v in tot.items()} | dict(epoch=epoch))
    for k, v in history[-1].items(): writer.add_scalar(f"epoch/{k}", v, step)
    print("epoch summary:", {k: round(v, 4) for k, v in history[-1].items()}, flush=True)
torch.save(dict(model_config=model.model_config, model_state_dict=model.state_dict(), epoch=epochs, global_step=step,
                history=history, vq_options=dict(normalize=normalize, entropy_weight=entropy_weight), stride=stride, arch=arch), f"{S}/{name}.pt")
print(f"elapsed {time.time()-t0:.1f}s; saved {S}/{name}.pt", flush=True)
