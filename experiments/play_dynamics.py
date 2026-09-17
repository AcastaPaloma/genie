"""Run the trained world model locally and write animated DOOM rollouts.

    python3 experiments/play_dynamics.py [--checkpoint PATH] [--clip N] [--context 4] [--steps 12]
        [--script "6x4,3x4,1x4"] [--maskgit-steps 25] [--temperature 1.0] [--out data/experiments/play]

Outputs:
  true_actions.gif / .png   top row: real frames; bottom row: generated from the same prompt with the
                            clip's own latent actions (what the LAM inferred), so drift is visible.
  scripted.gif / .png       generated from the same prompt with a scripted sequence of latent action
                            codes, e.g. "6x4,3x4,1x4" = code 6 for 4 steps, then code 3, then code 1.
Frames are the 90 content rows, shown at 2x. GIFs play at ~4 fps (real rate is 8.75 decisions/s).
"""
import argparse, sys, time
from pathlib import Path
_ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(_ROOT))
import numpy as np, torch
from PIL import Image, ImageDraw
import dynamics_model as dm
from experiments.vq_variants import StrideClipDataset
from experiments.render_dynamics import to_img, psnr, pan, W, H, PAD


def strip(rows, labels, col_labels):
    img = Image.new("RGB", (len(rows[0]) * W, len(rows) * (H + PAD) + PAD), "black"); d = ImageDraw.Draw(img)
    for c, l in enumerate(col_labels): d.text((c * W + 4, 2), l, fill="white")
    for r, row in enumerate(rows):
        for c, f in enumerate(row): img.paste(f, (c * W, PAD + r * (H + PAD)))
        d.text((4, PAD + r * (H + PAD) + 2), labels[r], fill="yellow")
    return img


def gif(path, rows, labels, fps=4):
    frames = []
    for j in range(len(rows[0])):
        img = Image.new("RGB", (W, len(rows) * (H + PAD)), "black"); d = ImageDraw.Draw(img)
        for r, row in enumerate(rows):
            img.paste(row[j], (0, r * (H + PAD) + PAD)); d.text((4, r * (H + PAD) + 2), f"{labels[r]}  frame {j}", fill="yellow")
        frames.append(img)
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=int(1000 / fps), loop=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=str(_ROOT / "data/checkpoints/dynamics_a100_epoch6_weights.pt"))
    ap.add_argument("--tokenizer", default=str(_ROOT / "data/checkpoints/vae_latest_best_weights.pt"))
    ap.add_argument("--lam", default=str(_ROOT / "data/checkpoints/lam_spatial_a100_pan.pt"))
    ap.add_argument("--clip", type=int, default=1); ap.add_argument("--context", type=int, default=4)
    ap.add_argument("--steps", type=int, default=12); ap.add_argument("--script", default="6x4,3x4,1x4")
    ap.add_argument("--maskgit-steps", type=int, default=25); ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--out", default=str(_ROOT / "data/experiments/play")); ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    torch.manual_seed(args.seed); device = dm._resolve_device(None); out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = dm.DynamicsModel(**ck["model_config"]); model.load_state_dict(ck["model_state_dict"]); model = model.to(device).eval()
    tok, lam = dm.load_tokenizer(args.tokenizer, device), dm.load_lam(args.lam, device)
    print(f"dynamics epoch {ck.get('epoch')} step {ck.get('global_step')} on {device}; LAM K={lam.K}", flush=True)
    T0, R = args.context, args.steps
    if T0 + R > 16: raise ValueError("context + steps must be <= 16 for the ground-truth comparison")
    sampling = dict(steps=args.maskgit_steps, temperature=args.temperature)
    ds = StrideClipDataset(_ROOT / "data/val", T=16, stride=ck.get("stride", 4), start_step=16, max_windows=args.clip + 1)
    video = ds[args.clip][0][None].to(device)                       # (1,16,96,160,3)
    with torch.no_grad():
        ids, actions = dm.encode_clips(tok, lam, video)              # (1,16,960), (1,15,32)
        _, code_ids, _ = lam.encode(video)
        # 1. the clip's own latent actions vs ground truth
        t0 = time.time()
        gen = dm.rollout(model, ids[:, :T0], actions[:, :T0 - 1 + R], R, **sampling)
        gen_px = tok.decode_tokens(torch.cat([ids[:, :T0], gen], 1))[0]   # (T0+R,96,160,3)
        print(f"true-action rollout: {R} frames in {time.time()-t0:.0f}s; inferred codes {code_ids[0, :T0-1+R].tolist()}", flush=True)
        truth = video[0]
        p = psnr(gen_px[T0:], truth[T0:T0 + R]); c = psnr(truth[T0 - 1:T0].expand(R, -1, -1, -1), truth[T0:T0 + R])
        print("  PSNR per step: " + " ".join(f"{v:.1f}" for v in p.tolist()) + "  (copy-last " + " ".join(f"{v:.1f}" for v in c.tolist()) + ")", flush=True)
        rows = [[to_img(truth[j]) for j in range(T0 + R)], [to_img(gen_px[j]) for j in range(T0 + R)]]
        labels = ["real", "generated (own actions)"]
        cols = [f"frame {j}" + (" (prompt)" if j < T0 else "") for j in range(T0 + R)]
        strip(rows, labels, cols).save(out / "true_actions.png"); gif(out / "true_actions.gif", rows, labels)
        # 2. scripted actions
        script = []
        for part in args.script.split(","):
            code, n = part.split("x"); script += [int(code)] * int(n)
        S = len(script)
        acts = torch.cat([actions[:, :T0 - 1], dm.action_vectors(lam, torch.tensor(script, device=device))[None]], 1)
        t0 = time.time()
        gen2 = dm.rollout(model, ids[:, :T0], acts, S, **sampling)
        gen2_px = tok.decode_tokens(torch.cat([ids[:, :T0], gen2], 1))[0]
        pans = pan(gen2_px[T0 - 1:T0 + S - 1], gen2_px[T0:T0 + S])
        print(f"scripted rollout {args.script}: {S} frames in {time.time()-t0:.0f}s; per-step pan (px, + = scene moves right): " + " ".join(f"{v:+.0f}" for v in pans.tolist()), flush=True)
        rows2 = [[to_img(gen2_px[j]) for j in range(T0 + S)]]
        cols2 = [f"frame {j} (prompt)" for j in range(T0)] + [f"code {k}" for k in script]
        strip(rows2, ["generated (scripted)"], cols2).save(out / "scripted.png"); gif(out / "scripted.gif", rows2, ["scripted: " + args.script])
    print(f"wrote {out}/true_actions.gif true_actions.png scripted.gif scripted.png", flush=True)


if __name__ == "__main__":
    main()
