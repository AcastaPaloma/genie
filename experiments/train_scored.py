"""LAM training with prediction-based scoring after every epoch.

Recipe: stride-4 frames, change-in-pooled-features encoder (V4), data-init + threshold-restart VQ.
Scores on a fixed val set:  codebook perplexity;  dPSNR (own vs random actions);
consistency across contexts (same code applied to many scenes -> same pan direction?);
distinctness between codes;  grouped latent->real mapping accuracy (fit on half, test on half).
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

import json, math, sys, time
from pathlib import Path
pass
S = str(_EXP); sys.path.insert(0, S)
from dataclasses import replace
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from config import LAM_TRAINING, build_optimizer
from lam import LAM
from vq_variants import VQ, StrideClipDataset
from arch_variants import apply_arch

name = sys.argv[1] if len(sys.argv) > 1 else "scored_v4"
epochs = int(sys.argv[2]) if len(sys.argv) > 2 else 12
ARCH = sys.argv[3] if len(sys.argv) > 3 else "diffenc"
ROOT = Path(str(_ROOT)); CKPT = ROOT / "data/checkpoints" / f"lam_{name}.pt"
CKPT.parent.mkdir(parents=True, exist_ok=True)
torch.manual_seed(0); device = torch.device(_DEV); STRIDE = 4
torch.backends.cuda.matmul.allow_tf32 = True; torch.backends.cudnn.allow_tf32 = True
AMP = _DEV == "cuda"   # bf16 autocast on CUDA, as in the paper; VQ distances stay fp32 inside the codebook

def soft_pan(ctx, pred, max_shift=14, temp=0.05):
    """Differentiable horizontal pan of pred relative to ctx, in pixels. Both (N,H,W,3) in [0,1].
    Soft argmin over integer shifts of the squared error between shifted ctx and pred."""
    a = ctx[..., :90, :, :].float().mean(-1)[..., 8:72, :]; b = pred[..., :90, :, :].float().mean(-1)[..., 8:72, :]
    shifts = torch.arange(-max_shift, max_shift + 1, device=a.device, dtype=torch.float32)
    errs = []
    for s_ in range(-max_shift, max_shift + 1):
        d = (a[..., :, s_:] - b[..., :, :160 - s_]) if s_ >= 0 else (a[..., :, :160 + s_] - b[..., :, -s_:])
        errs.append((d ** 2).mean(dim=(-2, -1)))
    errs = torch.stack(errs, -1)                                   # (N, 29)
    w = torch.softmax(-errs / (temp * errs.mean(-1, keepdim=True).detach().clamp(min=1e-8)), dim=-1)
    return (w * shifts).sum(-1)                                     # (N,)


def consistency_loss(ctx, pred, codes, K, max_samples=128):
    """How badly the code explains the predicted pan across different scenes (1 - R^2).
    Small when knowing the code tells you how the picture panned, regardless of scene."""
    ctx = ctx.flatten(0, 1); pred = pred.flatten(0, 1); codes = codes.flatten()
    if ctx.shape[0] > max_samples:
        pick = torch.randperm(ctx.shape[0], device=ctx.device)[:max_samples]; ctx, pred, codes = ctx[pick], pred[pick], codes[pick]
    p = soft_pan(ctx, pred); mean = p.mean(); between = p.new_zeros(())
    for k in range(K):
        m = codes == k
        if m.sum() >= 2: between = between + m.sum() * (p[m].mean() - mean) ** 2
    between = between / p.shape[0]                       # variance of pan explained by the code
    total = p.var(unbiased=False)
    # 1 - R^2: 1 when the code explains none of the pan variance (or nothing pans), 0 when it explains all of it.
    # Scale-free, index-free (restarts do not disturb it), and never satisfied by not panning.
    return 1.0 - between / (total + 0.5), p


def main():
    # ---------------- fixed evaluation set ----------------
    val = StrideClipDataset(ROOT / "data/val", stride=STRIDE, start_step=16, max_windows=128)
    V, R = zip(*[val[i] for i in range(len(val))]); V = torch.stack(V); R = torch.stack(R)      # (128,16,96,160,3), (128,15)
    gray = V[..., :90, :, :].mean(-1); band = gray[:, :, 8:72, :]
    def pan(a, b, max_shift=14):
        """Horizontal shift of scene content from a to b, in pixels (+ = content moved right)."""
        best = None; best_err = None
        for s in range(-max_shift, max_shift + 1):
            err = (((a[..., :, s:] - b[..., :, :160 - s]) if s >= 0 else (a[..., :, :160 + s] - b[..., :, -s:])) ** 2).mean(dim=(-2, -1))
            if best is None: best, best_err = torch.full_like(err, float(s)), err; continue
            better = err < best_err; best = torch.where(better, torch.full_like(err, float(s)), best); best_err = torch.minimum(err, best_err)
        return best
    true_pan = pan(band[:, :-1], band[:, 1:])                                  # (128,15)
    def pan_class(p): return torch.where(p <= -2, 0, torch.where(p >= 2, 2, 1))    # 0 left, 1 none, 2 right
    # group the 18 real ids by what they do on screen (mean pan over the val set)
    ids = sorted(torch.unique(R).tolist()); id_group = {}
    for a in ids:
        m = R == a; mp = true_pan[m].mean().item(); id_group[a] = 0 if mp <= -2 else (2 if mp >= 2 else 1)
    GROUP = torch.tensor([id_group[a] for a in range(max(ids) + 1)])
    real_group = GROUP[R]                                                      # (128,15) in {0,1,2}
    print("real id -> visual group (0 left, 1 none, 2 right):", {a: id_group[a] for a in ids}, flush=True)
    def psnr(a, b):
        mse = ((a[:, :, :90] - b[:, :, :90]) ** 2).flatten(2).mean(-1); return 10 * torch.log10(1.0 / mse.clamp(min=1e-10))
    def nmi(a, b):
        a = a.flatten().long(); b = b.flatten().long(); C = torch.zeros(int(a.max()) + 1, int(b.max()) + 1)
        C.index_put_((a, b), torch.ones(a.shape[0]), accumulate=True); pj = C / C.sum(); pa = pj.sum(1); pb = pj.sum(0)
        mi = (pj * (pj / (pa[:, None] * pb[None, :] + 1e-12) + 1e-12).log()).sum()
        ha = -(pa * (pa + 1e-12).log()).sum(); hb = -(pb * (pb + 1e-12).log()).sum(); return (mi / max(1e-9, math.sqrt(ha * hb))).item()

    @torch.no_grad()
    def score(model, step, writer, log):
        model.eval(); K = model.K; B = 16; lookup = model.cookbook.lookup
        g = torch.Generator().manual_seed(123)
        idx_all, p_own, p_rand = [], [], []
        for i in range(0, len(V), B):
            v = V[i:i + B].to(device); z_q, idx, _ = model.encode(v); target = v[:, 1:]
            idx_all.append(idx.cpu())
            p_own.append(psnr(model.decoder(v[:, :-1], z_q), target).cpu())
            rand = torch.randint(0, K, idx.shape, generator=g).to(device)
            p_rand.append(psnr(model.decoder(v[:, :-1], lookup(rand)), target).cpu())
        idx_all = torch.cat(idx_all); own = torch.cat(p_own).mean().item(); rnd = torch.cat(p_rand).mean().item()
        counts = torch.bincount(idx_all.flatten(), minlength=K).float(); p = counts / counts.sum()
        ppl = torch.exp(-(p * (p + 1e-10).log()).sum()).item()
        # --- consistency across contexts: 64 contexts x K codes, 4 context frames, code applied at the last step ---
        t = 3; ctx = V[:64, :t + 1]; pans = torch.zeros(K, 64)
        for i in range(0, 64, B):
            c = ctx[i:i + B].to(device); _, cidx, _ = model.encode(c)                 # (B,3) inferred context actions
            base = torch.cat([cidx, torch.zeros_like(cidx[:, :1])], 1)             # (B,4); last slot is the probe code
            for k in range(K):
                ids_k = base.clone(); ids_k[:, t] = k
                pred = model.decoder(c, lookup(ids_k))[:, t].cpu()                     # predicted frame t+1
                pans[k, i:i + B] = pan(band[i:i + B, t], pred[..., :90, :, :].mean(-1)[:, 8:72, :])
        cls = pan_class(pans)                                                          # (K,64)
        rows = []
        for k in range(K):
            hist = torch.bincount(cls[k], minlength=3).float() / 64; maj = int(hist.argmax())
            rows.append(dict(code=k, used=int(counts[k]), mean_pan=pans[k].mean().item(), std_pan=pans[k].std().item(),
                             majority=["left", "none", "right"][maj], agreement=hist.max().item()))
        used = [r for r in rows if r["used"] > 0]
        consistency = sum(r["agreement"] for r in used) / max(1, len(used))
        distinct = len({r["majority"] for r in used})
        # --- grouped latent -> real mapping: fit on first half of clips, test on second half ---
        A = slice(0, 64), slice(64, 128); la, lb = idx_all[A[0]].flatten(), idx_all[A[1]].flatten(); ga, gb = real_group[A[0]].flatten(), real_group[A[1]].flatten()
        table = torch.zeros(K, 3); table.index_put_((la, ga), torch.ones(la.shape[0]), accumulate=True)
        code_to_group = table.argmax(1); acc = (code_to_group[lb] == gb).float().mean().item()
        chance = torch.bincount(gb, minlength=3).float().max().item() / gb.numel()
        m = dict(step=step, perplexity=ppl, psnr_own=own, psnr_random=rnd, dpsnr=own - rnd, consistency=consistency,
                 distinct_directions=distinct, mapping_acc=acc, mapping_chance=chance, nmi_group=nmi(idx_all, real_group), codes=rows)
        nice = dict(perplexity="val_codebook_perplexity_of_8", psnr_own="psnr_with_own_actions_dB", psnr_random="psnr_with_random_actions_dB",
                    dpsnr="dPSNR_controllability_dB_higher_better", consistency="same_code_same_direction_across_scenes_chance0.4",
                    distinct_directions="distinct_directions_among_codes_max3", mapping_acc="latent_to_real_action_accuracy",
                    mapping_chance="latent_to_real_action_CHANCE", nmi_group="nmi_code_vs_real_left_none_right")
        for k_, v_ in m.items():
            if isinstance(v_, (int, float)): writer.add_scalar(f"eval/{nice.get(k_, k_)}", v_, step)
        for r in rows:
            writer.add_scalar(f"probe_same_code_on_64_scenes/mean_pan_px_code{r['code']}", r["mean_pan"], step)
            writer.add_scalar(f"probe_same_code_on_64_scenes/direction_agreement_code{r['code']}", r["agreement"], step)
        print(f"\n[eval @ step {step}] ppl={ppl:.2f}  PSNR own={own:.2f} random={rnd:.2f} dPSNR={own-rnd:+.2f}  "
              f"consistency={consistency:.2f} distinct_dirs={distinct}  map_acc={acc:.3f} (chance {chance:.3f})  nmi={m['nmi_group']:.3f}", flush=True)
        print("  code  used   mean_pan  std   majority  agreement")
        for r in rows: print(f"  {r['code']:>4}  {r['used']:>4}  {r['mean_pan']:+7.2f}  {r['std_pan']:5.2f}  {r['majority']:>8}  {r['agreement']:.2f}", flush=True)
        log.append(m); model.train(); return m

    # ---------------- model + data ----------------
    ds = StrideClipDataset(ROOT / "data/train", stride=STRIDE, start_step=8)
    BATCH = int(_os.environ.get("BATCH", 16))
    loader = DataLoader(ds, batch_size=BATCH, shuffle=True, num_workers=_WORKERS, generator=torch.Generator().manual_seed(0),
                        **({"multiprocessing_context": "spawn", "persistent_workers": True} if _WORKERS else {}))  # spawn: forked ArrayRecord readers segfault
    WIDTH, BLOCKS, HEADS = (int(_os.environ.get(k, d)) for k, d in (("WIDTH", 256), ("BLOCKS", 4), ("HEADS", 4)))
    model = apply_arch(LAM(d_width=WIDTH, st_blocks=BLOCKS, st_heads=HEADS, head_dim=64), ARCH)
    model.cookbook = VQ(K=model.K, code_width=32); model = model.to(device)
    PANHEAD = float(_os.environ.get("PANHEAD", 0.0))             # weight of the true-pan head on the quantized code (0 = off)
    ROUNDTRIP = float(_os.environ.get("ROUNDTRIP", 0.0))         # weight of the code round-trip loss (0 = off)
    CONSISTENCY = float(_os.environ.get("CONSISTENCY", 0.0))   # weight of the within-code pan-variance loss (0 = off)
    LR, MIN_LR, WARMUP = float(_os.environ.get("LR", 3e-4)), float(_os.environ.get("MIN_LR", 3e-4)), int(_os.environ.get("WARMUP", 100))
    cfg = replace(LAM_TRAINING, max_lr=LR, min_lr=MIN_LR, warmup_steps=WARMUP, steps=epochs * len(loader), global_batch_size=BATCH)  # warmup then cosine to MIN_LR
    pan_head = torch.nn.Linear(model.cookbook.code_width, 1).to(device)   # reads the code vector, must predict the TRUE pan
    opt, sched = build_optimizer(torch.nn.ModuleList([model, pan_head]), cfg); writer = SummaryWriter(f"{S}/tb/{name}")
    print(f"{name}: arch={ARCH} width={WIDTH} blocks={BLOCKS} heads={HEADS} batch={BATCH} lr={LR}->{MIN_LR} warmup={WARMUP} consistency={CONSISTENCY} roundtrip={ROUNDTRIP} panhead={PANHEAD} device={_DEV} workers={_WORKERS}, {len(ds)} train windows, {len(loader)} batches/epoch, {epochs} epochs, eval set {len(V)} clips", flush=True)
    history, evals, step, t0 = [], [], 0, time.time()
    score(model, 0, writer, evals)
    for epoch in range(1, epochs + 1):
        model.train(); tot = dict(loss=0., reconstruction_loss=0., vq_loss=0., perplexity=0.); n = 0
        for b, (video, _) in enumerate(loader, 1):
            video = video.to(device); opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=_DEV, dtype=torch.bfloat16, enabled=AMP):
                r = model(video, return_details=True)
            loss = r["loss"]
            if CONSISTENCY > 0:
                c_loss, _ = consistency_loss(video[:, :-1], r["prediction"], r["indices"], model.K)
                loss = loss + CONSISTENCY * c_loss; writer.add_scalar("loss/consistency_pan_var", c_loss.item(), step)
            if PANHEAD > 0:
                # Anchor: the code must determine the pan measured between the REAL frames. The decoder cannot
                # game this; gradient reaches the encoder through the straight-through estimator, so the
                # encoder sorts transitions into codes by how the camera actually turned.
                with torch.no_grad():
                    tp = soft_pan(video[:, :-1].flatten(0, 1), video[:, 1:].flatten(0, 1)) / 10.0    # true pan, scaled to ~unit
                z_sel = r["actions"].flatten(0, 1).float()
                ph_loss = torch.nn.functional.mse_loss(pan_head(z_sel).squeeze(-1), tp)
                loss = loss + PANHEAD * ph_loss
                writer.add_scalar("loss/panhead_mse", ph_loss.item(), step)
                writer.add_scalar("codebook/pan_r2_true", (1 - ph_loss / tp.var().clamp(min=1e-6)).item(), step)   # how much of the TRUE pan the code explains
            if ROUNDTRIP > 0:
                # Round trip: the code must be recoverable from (context frame, predicted frame).
                # Gradient reaches the decoder (make each code's effect recognizable) and the encoder
                # (recognize it); the codebook rows are detached so this never moves the codes themselves.
                ctx_f = video[:, :-1].flatten(0, 1); pr_f = r["prediction"].flatten(0, 1); tgt = r["indices"].flatten()
                pick = torch.randperm(ctx_f.shape[0], device=ctx_f.device)[:128]
                pairs = torch.stack([ctx_f[pick], pr_f[pick]], 1)                       # (n, 2, H, W, 3)
                z_rt = model.encoder(pairs)[:, 1].float()                                # code vector for the 0->1 change
                cb = model.cookbook.codes().detach().float()
                d = z_rt.square().sum(-1, keepdim=True) + cb.square().sum(-1) - 2 * z_rt @ cb.T
                logits = -d / (0.05 * d.mean().detach().clamp(min=1e-8))   # sharp: mean-normalized distances differ by fractions, so a plain 1/mean temperature left the softmax uniform
                rt_loss = torch.nn.functional.cross_entropy(logits, tgt[pick])
                rt_acc = (logits.argmax(-1) == tgt[pick]).float().mean()
                loss = loss + ROUNDTRIP * rt_loss
                writer.add_scalar("loss/roundtrip_ce", rt_loss.item(), step); writer.add_scalar("codebook/roundtrip_acc", rt_acc.item(), step)
            loss.backward(); opt.step(); sched.step(); step += 1; n += video.shape[0]
            for k in tot: tot[k] += r[k].detach().float().item() * video.shape[0]
            for tag, key in (("loss/total", "loss"), ("loss/reconstruction", "reconstruction_loss"), ("loss/vq", "vq_loss"), ("codebook/perplexity_batch", "perplexity")):
                writer.add_scalar(tag, r[key].item(), step)
            if b % 50 == 0 or b == len(loader):
                extra = (f" cons={c_loss.item():.2f}" if CONSISTENCY > 0 else "") + (f" rt_acc={rt_acc.item():.2f}" if ROUNDTRIP > 0 else "") + (f" pan_r2={(1 - ph_loss / tp.var().clamp(min=1e-6)).item():.2f}" if PANHEAD > 0 else "")
                print(f"epoch {epoch}/{epochs} batch {b}/{len(loader)} loss={r['loss'].item():.5f} recon={r['reconstruction_loss'].item():.5f} vq={r['vq_loss'].item():.5f} ppl={r['perplexity'].item():.2f}{extra}", flush=True)
        history.append({k: v / n for k, v in tot.items()} | dict(epoch=epoch, step=step))
        score(model, step, writer, evals)
        torch.save(dict(model_config=model.model_config, model_state_dict=model.state_dict(), vq_options=dict(normalize=False, entropy_weight=0.0),
                        arch=ARCH, stride=STRIDE, epoch=epoch, global_step=step, history=history, evals=evals), CKPT)
        json.dump(evals, open(f"{S}/{name}_evals.json", "w"), indent=1)
        print(f"Finished epoch {epoch} ({time.time()-t0:.0f}s elapsed); saved {CKPT}", flush=True)
    print(f"elapsed {time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
