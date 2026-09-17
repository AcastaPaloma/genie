"""Render what a dynamics checkpoint actually predicts, as PNG grids plus PSNR numbers.

    python3 experiments/render_dynamics.py data/checkpoints/dynamics_a100.pt data/experiments/dynamics_a100_render \
        [--clips 6] [--context 4] [--rollout 8] [--tokenizer PATH] [--lam PATH] [--val-dir DIR]

Writes OUT_DIR/epoch{E}/next_frame.png (last context frame | truth | one-pass argmax | MaskGIT),
actions.png (same context, MaskGIT next frame under each latent action, with its screen pan),
rollout.png (true frames vs generated frames driven by the clip's own LAM actions) and metrics.json.
"""
import argparse, json, math, sys, time
from pathlib import Path
_ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(_ROOT))
import numpy as np, torch
from PIL import Image, ImageDraw
import dynamics_model as dm
from experiments.vq_variants import StrideClipDataset

W, H, PAD = 320, 180, 14


def to_img(frame):  # (96,160,3) float [0,1] -> PIL, cropped to the 90 content rows, 2x nearest
    a = (frame[:90].clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
    return Image.fromarray(a).resize((W, H), Image.NEAREST)


def grid(rows, col_labels, row_labels=None):
    cols = max(len(r) for r in rows)
    img = Image.new("RGB", (cols * W, len(rows) * (H + PAD) + PAD), "black"); draw = ImageDraw.Draw(img)
    for c, label in enumerate(col_labels):
        draw.text((c * W + 4, 2), label, fill="white")
    for r, row in enumerate(rows):
        for c, frame in enumerate(row):
            img.paste(frame, (c * W, PAD + r * (H + PAD)))
        if row_labels:
            draw.text((4, PAD + r * (H + PAD) + 2), row_labels[r], fill="yellow")
    return img


def psnr(a, b):  # (..., 96,160,3) -> per-frame dB over the content rows
    mse = ((a[..., :90, :, :] - b[..., :90, :, :]) ** 2).flatten(-3).mean(-1)
    return 10 * torch.log10(1.0 / mse.clamp(min=1e-10))


def mse(a, b):  # (..., 96,160,3) -> per-frame pixel MSE over the content rows, pixels in [0,1]
    return ((a[..., :90, :, :] - b[..., :90, :, :]) ** 2).flatten(-3).mean(-1)


def with_mse(metrics):
    """Add mse_* next to every psnr_* entry (PSNR = 10 log10(1 / MSE))."""
    for key in [k for k in metrics if k.startswith("psnr_")]:
        value = metrics[key]
        metrics["mse_" + key[5:]] = [10 ** (-v / 10) for v in value] if isinstance(value, list) else 10 ** (-value / 10)
    return metrics


def pan(a, b, max_shift=14):
    """Horizontal shift of scene content from frame a to b in pixels (+ = content moved right)."""
    ga, gb = a[..., :90, :, :].mean(-1)[..., 8:72, :], b[..., :90, :, :].mean(-1)[..., 8:72, :]
    best, best_err = None, None
    for s in range(-max_shift, max_shift + 1):
        err = (((ga[..., :, s:] - gb[..., :, :160 - s]) if s >= 0 else (ga[..., :, :160 + s] - gb[..., :, -s:])) ** 2).mean(dim=(-2, -1))
        if best is None:
            best, best_err = torch.full_like(err, float(s)), err; continue
        better = err < best_err
        best = torch.where(better, torch.full_like(err, float(s)), best); best_err = torch.minimum(err, best_err)
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint"); ap.add_argument("out_dir")
    ap.add_argument("--clips", type=int, default=6); ap.add_argument("--context", type=int, default=4)
    ap.add_argument("--rollout", type=int, default=8); ap.add_argument("--tokenizer", default=None)
    ap.add_argument("--lam", default=None); ap.add_argument("--val-dir", default=None)
    ap.add_argument("--steps", type=int, default=None); ap.add_argument("--temperature", type=float, default=None)
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--device", default=None)
    ap.add_argument("--pan-clips", type=int, default=24, help="clips for the per-code pan statistics (argmax decodes)")
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    device = dm._resolve_device(args.device)
    ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = dm.DynamicsModel(**ck["model_config"]); model.load_state_dict(ck["model_state_dict"]); model = model.to(device).eval()
    tok_path = args.tokenizer or str(_ROOT / ck["tokenizer_checkpoint"]) if not Path(ck["tokenizer_checkpoint"]).is_absolute() else ck["tokenizer_checkpoint"]
    lam_path = args.lam or str(_ROOT / ck["lam_checkpoint"]) if not Path(ck["lam_checkpoint"]).is_absolute() else ck["lam_checkpoint"]
    tokenizer, lam = dm.load_tokenizer(tok_path, device), dm.load_lam(lam_path, device)
    sampling = {k: v for k, v in dict(steps=args.steps, temperature=args.temperature).items() if v is not None}
    out = Path(args.out_dir) / f"epoch{ck.get('epoch', 0)}_step{ck.get('global_step', 0)}"; out.mkdir(parents=True, exist_ok=True)
    T0, R = args.context, args.rollout
    if T0 + R > 16 or T0 < 1 or R < 1:
        raise ValueError("context + rollout must fit in the 16-frame clips")
    print(f"dynamics epoch {ck.get('epoch')}{' (partial ' + str(ck['partial_epoch']) + ')' if ck.get('partial_epoch') else ''} step {ck.get('global_step')}; tokenizer {Path(tok_path).name}; LAM {Path(lam_path).name} (K={lam.K}); device {device}", flush=True)

    ds = StrideClipDataset(args.val_dir or _ROOT / "data/val", T=16, stride=ck.get("stride", 4), start_step=16, max_windows=args.clips)
    video = torch.stack([ds[i][0] for i in range(len(ds))]).to(device)          # (n,16,96,160,3)
    with torch.no_grad():
        ids, actions = dm.encode_clips(tokenizer, lam, video)                   # (n,16,960), (n,15,32)
        recon = tokenizer.decode_tokens(ids)                                    # tokenizer ceiling
    n = len(ds); t0 = time.time()

    def decode_last(prefix_ids, new_frame):  # decode context + new frame, return the new frame's pixels
        return tokenizer.decode_tokens(torch.cat([prefix_ids, new_frame[:, None]], 1))[:, -1]

    # ---- 1. next frame: one-pass argmax vs MaskGIT ----
    with torch.no_grad():
        frame_mask = torch.zeros(n, T0 + 1, ids.shape[2], dtype=torch.bool, device=device); frame_mask[:, -1] = True
        argmax_ids = model(ids[:, :T0 + 1], actions[:, :T0], mask=frame_mask)[:, -1].argmax(-1)
        argmax_px = decode_last(ids[:, :T0], argmax_ids)
        gen_ids = dm.sample_next_frame(model, ids[:, :T0], actions[:, :T0], **sampling)
        gen_px = decode_last(ids[:, :T0], gen_ids)
    truth = video[:, T0]
    m = dict(next_frame=with_mse(dict(
        psnr_tokenizer_ceiling=psnr(recon[:, T0], truth).mean().item(),
        psnr_argmax=psnr(argmax_px, truth).mean().item(), psnr_maskgit=psnr(gen_px, truth).mean().item(),
        psnr_copy_last_context=psnr(video[:, T0 - 1], truth).mean().item(),
        token_acc_argmax=(argmax_ids == ids[:, T0]).float().mean().item(),
        token_acc_maskgit=(gen_ids == ids[:, T0]).float().mean().item())))
    rows = [[to_img(video[i, T0 - 1]), to_img(truth[i]), to_img(argmax_px[i]), to_img(gen_px[i])] for i in range(n)]
    grid(rows, [f"context frame {T0 - 1}", f"true frame {T0}", "one-pass argmax", f"MaskGIT {sampling.get('steps', dm.MASKGIT_STEPS)} steps"]).save(out / "next_frame.png")
    print(f"next frame: {json.dumps({k: round(v, 3) for k, v in m['next_frame'].items()})}  [{time.time()-t0:.0f}s]", flush=True)

    # ---- 2. same context, every latent action ----
    # Pan statistics come from deterministic argmax decodes over pan_clips clips: the
    # shift estimator is heavy-tailed (std ~9 px on generated frames), so a handful of
    # temperature-2 samples reads as a fake bias. Images still show MaskGIT samples.
    pan_ds = StrideClipDataset(args.val_dir or _ROOT / "data/val", T=16, stride=ck.get("stride", 4), start_step=16, max_windows=args.pan_clips)
    pan_video = torch.stack([pan_ds[i][0] for i in range(len(pan_ds))]).to(device)
    with torch.no_grad():
        pan_ids, pan_actions = dm.encode_clips(tokenizer, lam, pan_video)
    pan_ctx, pan_truth = pan_video[:, T0 - 1], pan_video[:, T0]
    pan_mask = torch.zeros(len(pan_ds), T0 + 1, pan_ids.shape[2], dtype=torch.bool, device=device); pan_mask[:, -1] = True
    per_code, code_rows = {}, [[to_img(video[i, T0 - 1])] for i in range(n)]
    with torch.no_grad():
        for k in range(lam.K):
            acts = pan_actions[:, :T0].clone()
            acts[:, -1] = dm.action_vectors(lam, torch.full((len(pan_ds),), k, device=device, dtype=torch.long))
            code_ids = model(pan_ids[:, :T0 + 1], acts, mask=pan_mask)[:, -1].argmax(-1)
            p = pan(pan_ctx, decode_last(pan_ids[:, :T0], code_ids))
            per_code[k] = dict(mean_pan=p.mean().item(), median_pan=p.median().item(), std_pan=p.std().item() if len(pan_ds) > 1 else 0.0)
            acts_img = actions[:, :T0].clone()
            acts_img[:, -1] = dm.action_vectors(lam, torch.full((n,), k, device=device, dtype=torch.long))
            px = decode_last(ids[:, :T0], dm.sample_next_frame(model, ids[:, :T0], acts_img, **sampling))
            for i in range(n):
                code_rows[i].append(to_img(px[i]))
    true_pan = pan(pan_ctx, pan_truth)
    m["actions"] = dict(true_pan_mean=true_pan.mean().item(), per_code=per_code, pan_clips=len(pan_ds), pan_method="argmax",
                        pan_spread_across_codes=float(np.std([v["mean_pan"] for v in per_code.values()])))
    labels = [f"context frame {T0 - 1}"] + [f"code {k}: pan {per_code[k]['mean_pan']:+.1f}px (argmax, n={len(pan_ds)})" for k in range(lam.K)]
    grid(code_rows, labels).save(out / "actions.png")
    print("actions (argmax pan over %d clips): " % len(pan_ds) + "  ".join(f"code{k} pan={v['mean_pan']:+.1f}±{v['std_pan']:.1f}" for k, v in per_code.items()) + f"  (true pan {true_pan.mean().item():+.1f}, spread across codes {m['actions']['pan_spread_across_codes']:.2f})  [{time.time()-t0:.0f}s]", flush=True)
    del pan_video, pan_ids, pan_actions

    # ---- 3. rollout with the clip's own LAM actions ----
    with torch.no_grad():
        gen = dm.rollout(model, ids[:, :T0], actions[:, :T0 - 1 + R], R, **sampling)   # (n,R,960)
        gen_px = tokenizer.decode_tokens(torch.cat([ids[:, :T0], gen], 1))[:, T0:]      # (n,R,96,160,3)
    step_psnr = psnr(gen_px, video[:, T0:T0 + R]).mean(0).tolist()
    m["rollout"] = with_mse(dict(psnr_per_step=step_psnr, psnr_copy_last_context=psnr(video[:, T0 - 1:T0].expand(-1, R, -1, -1, -1), video[:, T0:T0 + R]).mean(0).tolist()))
    rows, row_labels = [], []
    for i in range(min(n, 3)):
        rows.append([to_img(video[i, j]) for j in range(T0 + R)]); row_labels.append(f"clip {i} true")
        rows.append([to_img(video[i, j]) for j in range(T0)] + [to_img(gen_px[i, j]) for j in range(R)]); row_labels.append(f"clip {i} generated")
    grid(rows, [f"frame {j}" + (" (ctx)" if j < T0 else "") for j in range(T0 + R)], row_labels).save(out / "rollout.png")
    print("rollout PSNR per step: " + " ".join(f"{v:.1f}" for v in step_psnr) + f"  (copy-last baseline {' '.join(f'{v:.1f}' for v in m['rollout']['psnr_copy_last_context'])})  [{time.time()-t0:.0f}s]", flush=True)
    m.update(epoch=ck.get("epoch"), global_step=ck.get("global_step"), clips=n, context_frames=T0, rollout_frames=R, sampling=sampling)
    json.dump(m, open(out / "metrics.json", "w"), indent=1)
    print(f"wrote {out}/next_frame.png actions.png rollout.png metrics.json", flush=True)


if __name__ == "__main__":
    main()
