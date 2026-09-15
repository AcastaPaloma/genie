"""Evaluate a trained LAM checkpoint on the DOOM val shard.

Metrics: next-frame PSNR vs copy-last-frame baseline, controllability delta
(true latent actions vs random ones, as in the Genie paper's dPSNR), latent
code usage, and agreement between the 8 latent codes and the real ViZDoom
actions stored in the records. Also writes PNG grids for visual inspection.
"""
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

import io, math, pickle, sys
from pathlib import Path
import numpy as np
import torch
from PIL import Image, ImageDraw

pass
from lam import LAM
sys.path.insert(0, str(Path(__file__).parent))
from vq_variants import VQ, StrideClipDataset
from dataloader import _VideoRecordUnpickler, prepare_video
from array_record.python.array_record_data_source import ArrayRecordDataSource

ckpt_path = sys.argv[1]
out_dir = Path(sys.argv[2]); out_dir.mkdir(parents=True, exist_ok=True)
n_windows = int(sys.argv[3]) if len(sys.argv) > 3 else 400
device = torch.device(_DEV if torch.backends.mps.is_available() else "cpu")
T = 16

from arch_variants import load_checkpoint
model, ckpt = load_checkpoint(ckpt_path, device)
stride = ckpt.get("stride", 1)
lookup = getattr(model.cookbook, "lookup", model.cookbook.cookbook)
print("loaded", ckpt_path, "| stride:", stride, "| vq:", ckpt.get("vq_options"), "| arch:", ckpt.get("arch", "base"), "| epochs trained:", ckpt["epoch"], "| steps:", ckpt["global_step"])
print("last epoch metrics:", {k: round(v, 4) if isinstance(v, float) else v for k, v in ckpt["history"][-1].items()})

# ---- val windows with real actions ----
ds = StrideClipDataset(str(_ROOT / "data/val"), stride=stride, start_step=16, max_windows=n_windows)
videos, real_actions = zip(*[ds[i] for i in range(len(ds))]); videos, real_actions = list(videos), list(real_actions)
videos = torch.stack(videos); real_actions = torch.stack(real_actions)
print(f"val windows: {tuple(videos.shape)}, real action ids: {sorted(torch.unique(real_actions).tolist())}")

def psnr(a, b):  # per-frame PSNR on [0,1] pixels, cropped to the 90 real rows
    mse = ((a[:, :, :90] - b[:, :, :90]) ** 2).flatten(2).mean(-1)
    return 10 * torch.log10(1.0 / mse.clamp(min=1e-10))

B = 16
rows = dict(true=[], random=[], copy=[]); rows_t4 = dict(true=[], random=[], copy=[])
counts = torch.zeros(model.K); latent_ids = []
g = torch.Generator().manual_seed(0)
with torch.no_grad():
    for i in range(0, len(videos), B):
        v = videos[i:i + B].to(device)
        z_q, idx, _ = model.encode(v)
        pred = model.decoder(v[:, :-1], z_q)
        rand_idx = torch.randint(0, model.K, idx.shape, generator=g).to(device)
        pred_rand = model.decoder(v[:, :-1], lookup(rand_idx))
        target = v[:, 1:]
        p_true, p_rand, p_copy = psnr(pred, target), psnr(pred_rand, target), psnr(v[:, :-1], target)
        for k, p in (("true", p_true), ("random", p_rand), ("copy", p_copy)):
            rows[k].append(p.mean(1).cpu()); rows_t4[k].append(p[:, 3].cpu())  # paper reports t=4
        counts += torch.bincount(idx.flatten().cpu(), minlength=model.K).float()
        latent_ids.append(idx.cpu())
latent_ids = torch.cat(latent_ids)
m = {k: torch.cat(v).mean().item() for k, v in rows.items()}
m4 = {k: torch.cat(v).mean().item() for k, v in rows_t4.items()}
p = counts / counts.sum(); ppl = torch.exp(-(p * (p + 1e-10).log()).sum()).item()

print("\n=== next-frame PSNR (dB), mean over 15 transitions ===")
print(f"  LAM with its own latent actions : {m['true']:.2f}")
print(f"  LAM with random latent actions  : {m['random']:.2f}")
print(f"  copy previous frame (no model)  : {m['copy']:.2f}")
print(f"  dPSNR (true - random), all t    : {m['true'] - m['random']:.2f}")
print(f"  dPSNR at t=4 (paper convention) : {m4['true'] - m4['random']:.2f}")
print(f"\n=== codebook usage on val ===\n  counts: {counts.long().tolist()}\n  perplexity: {ppl:.2f} / {model.K}")

# ---- latent vs real action agreement ----
real = real_actions.flatten(); lat = latent_ids.flatten()
real_vals = sorted(torch.unique(real).tolist()); col = {a: j for j, a in enumerate(real_vals)}
C = torch.zeros(model.K, len(real_vals), dtype=torch.long)
for l, r in zip(lat.tolist(), real.tolist()): C[l, col[r]] += 1
majority_acc = (C.max(1).values.sum() / C.sum()).item()
chance = (C.sum(0).max() / C.sum()).item()
pl = C.sum(1).float() / C.sum(); pr = C.sum(0).float() / C.sum(); pj = C.float() / C.sum()
mi = (pj * (pj / (pl[:, None] * pr[None, :]) + 1e-12).log()).nan_to_num().sum().item()
hl = -(pl * (pl + 1e-12).log()).sum().item(); hr = -(pr * (pr + 1e-12).log()).sum().item()
print("\n=== latent code vs real DOOM action ===")
print(f"  contingency rows=latent 0..{model.K-1}, cols=real ids {real_vals}")
for l in range(model.K): print(f"  L{l}: {C[l].tolist()}   majority real={real_vals[C[l].argmax().item()]}")
print(f"  map-each-latent-to-its-majority-real accuracy: {majority_acc:.3f}   chance (most common real): {chance:.3f}")
print(f"  normalized mutual information: {mi / max(1e-9, math.sqrt(hl * hr)):.3f}  (0 = independent, 1 = perfect)")

# ---- visual grids ----
def to_img(t):  # (H,W,3) float [0,1] -> PIL, cropped to 90 rows, 2x upscale
    a = (t[:90].clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
    return Image.fromarray(a).resize((320, 180), Image.NEAREST)
def diff_img(a, b):
    d = ((a[:90] - b[:90]).abs().mean(-1, keepdim=True).expand(-1, -1, 3) * 4).clamp(0, 1)
    return to_img(d)

with torch.no_grad():
    v = videos[:4].to(device)
    z_q, idx, _ = model.encode(v)
    pred = model.decoder(v[:, :-1], z_q)
    rand_idx = torch.randint(0, model.K, idx.shape, generator=g).to(device)
    pred_rand = model.decoder(v[:, :-1], lookup(rand_idx))
    # grid 1: per clip, one transition (t=4): input | prediction | truth | |pred-truth| | random-action prediction
    labels = ["frame t", "LAM pred t+1", "true t+1", "|pred - true| x4", "random-action pred"]
    W, H, pad = 320, 180, 24
    grid = Image.new("RGB", (5 * W, 4 * (H + pad)), "black"); dr = ImageDraw.Draw(grid)
    for r in range(4):
        t = 3; y = r * (H + pad)
        ims = [to_img(v[r, t]), to_img(pred[r, t]), to_img(v[r, t + 1]), diff_img(pred[r, t], v[r, t + 1]), to_img(pred_rand[r, t])]
        for c, im in enumerate(ims):
            grid.paste(im, (c * W, y + pad))
            dr.text((c * W + 4, y + 4), f"{labels[c]}  latent={idx[r, t].item()} real={real_actions[r, t].item()}" if c == 1 else labels[c], fill="white")
    grid.save(out_dir / "recon_grid.png")
    # grid 2: same context, every one of the K latent actions applied at the last step
    r, t = 0, 3
    ctx = v[r:r + 1, :t + 1]; base_idx = idx[r:r + 1, :t + 1].clone()
    grid2 = Image.new("RGB", ((model.K + 1) * W, 2 * (H + pad)), "black"); dr = ImageDraw.Draw(grid2)
    grid2.paste(to_img(v[r, t]), (0, pad)); dr.text((4, 4), "context frame t", fill="white")
    grid2.paste(to_img(v[r, t + 1]), (0, H + 2 * pad)); dr.text((4, H + pad + 4), f"true t+1 (latent {idx[r, t].item()})", fill="white")
    for a in range(model.K):
        ids = base_idx.clone(); ids[0, t] = a
        p_a = model.decoder(ctx, lookup(ids))[0, t]
        grid2.paste(to_img(p_a), ((a + 1) * W, pad)); dr.text(((a + 1) * W + 4, 4), f"action {a}", fill="white")
        grid2.paste(diff_img(p_a, v[r, t]), ((a + 1) * W, H + 2 * pad)); dr.text(((a + 1) * W + 4, H + pad + 4), f"|action {a} - frame t| x4", fill="white")
    grid2.save(out_dir / "all_actions_grid.png")
print(f"\nwrote {out_dir/'recon_grid.png'} and {out_dir/'all_actions_grid.png'}")
