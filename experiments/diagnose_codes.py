"""What do the latent codes correlate with? Estimate per-transition camera pan (horizontal
shift between frame t and t+1, from pixels), motion magnitude, and target brightness, then
measure agreement of the latent code with each, and with the real DOOM action IDs."""
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

import math, sys
from pathlib import Path
import numpy as np, torch
pass; sys.path.insert(0, str(Path(__file__).parent))
from lam import LAM
from vq_variants import VQ, StrideClipDataset

from arch_variants import load_checkpoint
model, ckpt = load_checkpoint(sys.argv[1], _DEV)
stride = ckpt.get("stride", 1)
ds = StrideClipDataset(str(_ROOT / "data/val"), stride=stride, start_step=16, max_windows=400)
videos, real = zip(*[ds[i] for i in range(len(ds))]); videos = torch.stack(videos); real = torch.stack(real)

with torch.no_grad():
    lat = torch.cat([model.encode(videos[i:i+16].to(_DEV))[1].cpu() for i in range(0, len(videos), 16)])

# ---- pixel-based motion estimates (grayscale, content rows only, HUD excluded) ----
g = videos[..., :90, :, :].mean(-1)                      # (N,T,90,160)
band = g[:, :, 8:72, :]                                  # skip ceiling edge + HUD
def pan(a, b, max_shift=14):                             # shift of scene content between frames, in pixels
    best, best_err = 0, None
    for s in range(-max_shift, max_shift + 1):
        if s >= 0: err = ((a[..., :, s:] - b[..., :, :160 - s]) ** 2).mean(dim=(-2, -1))
        else:      err = ((a[..., :, :160 + s] - b[..., :, -s:]) ** 2).mean(dim=(-2, -1))
        best_err = err if best_err is None else best_err
        if s == -max_shift: best = torch.full_like(err, float(s)); best_err = err; continue
        better = err < best_err; best = torch.where(better, torch.full_like(err, float(s)), best); best_err = torch.minimum(err, best_err)
    return best
dx = pan(band[:, :-1], band[:, 1:])                      # (N,T-1) scene shift in px; + = scene moved right = camera turned left
mag = (g[:, 1:] - g[:, :-1]).abs().mean(dim=(-2, -1))    # motion magnitude
bright = g[:, 1:].mean(dim=(-2, -1))                     # target-frame brightness

def nmi(a, b):
    a = a.flatten().long(); b = b.flatten().long()
    A = a.max() + 1; B = b.max() + 1
    C = torch.zeros(A, B); C.index_put_((a, b), torch.ones_like(a, dtype=torch.float), accumulate=True)
    pj = C / C.sum(); pa = pj.sum(1); pb = pj.sum(0)
    mi = (pj * (pj / (pa[:, None] * pb[None, :]) + 1e-12).log()).nan_to_num().sum()
    ha = -(pa * (pa + 1e-12).log()).sum(); hb = -(pb * (pb + 1e-12).log()).sum()
    return (mi / max(1e-9, math.sqrt(ha * hb))).item()
def qbins(x, n=4):
    edges = torch.quantile(x.flatten(), torch.linspace(0, 1, n + 1)[1:-1]); return torch.bucketize(x, edges)
pan_cls = torch.where(dx <= -3, 0, torch.where(dx >= 3, 2, 1))   # left / still / right (with deadzone)

print(f"model: {sys.argv[1].split('/')[-1]}  stride={stride}  transitions={lat.numel()}")
print("\n=== NMI of latent code with ... (0 = independent) ===")
for name, target in [("real DOOM action id (18 classes)", real), ("camera pan direction L/none/R", pan_cls),
                     ("camera pan, 4 quantile bins", qbins(dx)), ("motion magnitude, 4 bins", qbins(mag)),
                     ("target-frame brightness, 4 bins", qbins(bright))]:
    print(f"  {name:36s} {nmi(lat, target):.3f}")

print("\n=== per latent code: mean pan (px), mean motion, mean brightness, count ===")
for c in range(model.K):
    m = lat == c
    if m.sum() == 0: print(f"  L{c}: unused"); continue
    print(f"  L{c}: pan={dx[m].mean():+5.2f}  motion={mag[m].mean():.4f}  bright={bright[m].mean():.3f}  n={int(m.sum())}")

print("\n=== per real DOOM action id: mean pan (px), mean motion, count  (what the ids appear to do) ===")
for a in sorted(torch.unique(real).tolist()):
    m = real == a
    print(f"  id {a:2d}: pan={dx[m].mean():+5.2f}  motion={mag[m].mean():.4f}  n={int(m.sum()):5d}   latent histogram={torch.bincount(lat[m], minlength=model.K).tolist()}")
