"""Evaluate a saved tokenizer on a reproducible held-out sample and render previews."""
import argparse
import json
import math
from pathlib import Path
import random
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from PIL import Image, ImageDraw
import torch

from config import CONTENT_SIZE
from dataloader import ArrayRecordClipDataset
from tokenizer_validation import evaluate
from vae import VAE


def psnr(mse):
    return -10 * math.log10(max(mse, 1e-12))


def panel(original, reconstruction, caption):
    height, width = original.shape[:2]
    canvas = Image.new("RGB", (width * 4, height * 2 + 44), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 5), caption, fill="black")
    draw.text((8, 24), "Original", fill="black")
    draw.text((width * 2 + 8, 24), "Reconstruction", fill="black")
    for col, frame in enumerate((original, reconstruction)):
        pixels = (frame.clamp(0, 1).numpy() * 255).round().astype(np.uint8)
        canvas.paste(Image.fromarray(pixels).resize((width * 2, height * 2),
                     Image.Resampling.NEAREST), (col * width * 2, 44))
    return canvas


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-dir", default="data/test")
    parser.add_argument("--validation-dir", default="data/val")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--clips", type=int, default=256)
    parser.add_argument("--seed", type=int, default=20260914)
    parser.add_argument("--device", default="cuda:1")
    args = parser.parse_args()
    if args.clips < 4:
        parser.error("At least four clips are required")
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
        # Bound this evaluation while the separate DDP training job keeps running.
        torch.cuda.set_per_process_memory_fraction(0.22, device)
    saved = torch.load(args.checkpoint, map_location="cpu", weights_only=True, mmap=True)
    metadata = {key: saved.get(key) for key in
                ("global_step", "epoch", "next_sample", "dataset_size", "implementation_version")}
    model = VAE(**{"stabilize": False, **saved["model_config"]})
    model.load_state_dict(saved["model_state_dict"], strict=True)
    model = model.to(device).eval()
    del saved
    print(f"Checkpoint: {metadata}", flush=True)
    height, width = [min(a, b) for a, b in zip(CONTENT_SIZE, model.image_size)]
    dataset = ArrayRecordClipDataset(args.data_dir, T=model.temporal_pos.shape[0],
                                    content_size=(height, width), image_size=model.image_size)
    selected = sorted(random.Random(args.seed).sample(range(len(dataset)), min(args.clips, len(dataset))))
    preview_indices = [selected[int(i)] for i in np.linspace(0, len(selected)-1, 4)]
    counts = torch.zeros(model.cb.K, dtype=torch.long)
    records, previews = [], {}
    worst = None
    amp = device.type == "cuda"
    dtype = torch.bfloat16 if amp and torch.cuda.is_bf16_supported() else torch.float16
    started = time.perf_counter()
    for position, index in enumerate(selected):
        video = dataset[index][None].to(device)
        with torch.autocast(device.type, dtype=dtype, enabled=amp):
            result = model(video, return_details=True)
        prediction = result["reconstruction"].float()
        if not torch.isfinite(prediction).all():
            raise FloatingPointError(f"Nonfinite reconstruction at clip {index}")
        counts += torch.bincount(result["indices"].cpu().flatten(), minlength=model.cb.K)
        target = video[:, :, :height, :width]
        pred = prediction[:, :, :height, :width]
        error = pred - target
        mse = error.square().mean().item()
        mean_color = target.mean(dim=(2, 3), keepdim=True)
        row = dict(index=index, record_and_start=dataset.window_location(index),
                   mse=mse, psnr=psnr(mse), mae=error.abs().mean().item(),
                   padded_mse=(prediction-video).square().mean().item(),
                   mean_color_baseline_mse=(target-mean_color).square().mean().item(),
                   saturated_fraction=((pred < 1e-4) | (pred > 1-1e-4)).float().mean().item())
        records.append(row)
        if index in preview_indices or worst is None or mse > worst[0]:
            pair = (target[0].cpu(), pred[0].cpu())
            if index in preview_indices:
                previews[index] = pair
            if worst is None or mse > worst[0]:
                worst = (mse, index, pair)
        if (position + 1) % 32 == 0:
            print(f"Evaluated {position+1}/{len(selected)} clips; "
                  f"mean MSE={sum(r['mse'] for r in records)/len(records):.6f}", flush=True)
    elapsed = time.perf_counter() - started
    probabilities = counts.double() / counts.sum()
    nonzero = probabilities[probabilities > 0]
    mse = sum(r["mse"] for r in records) / len(records)
    metrics = dict(checkpoint=str(Path(args.checkpoint).resolve()), checkpoint_metadata=metadata,
                   split=str(Path(args.data_dir).resolve()), available_clips=len(dataset),
                   clips=len(selected), frames=len(selected)*dataset.T, seed=args.seed,
                   content_size=[height, width], padded_size=list(model.image_size),
                   content_mse=mse, content_psnr=psnr(mse),
                   mean_per_clip_psnr=sum(r["psnr"] for r in records)/len(records),
                   content_mae=sum(r["mae"] for r in records)/len(records),
                   padded_mse=sum(r["padded_mse"] for r in records)/len(records),
                   mean_color_baseline_mse=sum(r["mean_color_baseline_mse"] for r in records)/len(records),
                   active_codes=int((counts > 0).sum()), total_codes=model.cb.K,
                   code_perplexity=float(torch.exp(-(nonzero*nonzero.log()).sum())),
                   dominant_code_fraction=float(probabilities.max()),
                   saturated_fraction=sum(r["saturated_fraction"] for r in records)/len(records),
                   per_clip_psnr_percentiles=dict(zip(["p10", "p50", "p90"],
                      np.percentile([r["psnr"] for r in records], [10, 50, 90]).tolist())),
                   elapsed_seconds=elapsed,
                   timing_note="Evaluation shared a GPU with live training; not an isolated speed benchmark.",
                   cuda_peak_allocated_gib=torch.cuda.max_memory_allocated(device)/2**30 if amp else None,
                   fixed_preview_indices=preview_indices, worst_clip_index=worst[1],
                   code_counts=counts.tolist(), per_clip=records)
    # Match the trainer's four fixed validation clips for a like-for-like history comparison.
    validation = ArrayRecordClipDataset(args.validation_dir, T=dataset.T,
                  content_size=(height, width), image_size=model.image_size)
    val_indices = torch.linspace(0, len(validation)-1, 4).long().tolist()
    metrics["fixed_validation"] = evaluate(model, [validation[i] for i in val_indices], device,
                                step=metadata["global_step"], output_dir=output / "fixed_validation")
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    panels, animation = [], []
    for index in preview_indices:
        original, reconstructed = previews[index]
        quality = next(r["psnr"] for r in records if r["index"] == index)
        caption = f"Step {metadata['global_step']} | test clip {index} | PSNR {quality:.2f} dB"
        panels.append(panel(original[8], reconstructed[8], caption))
        animation.extend(panel(a, b, caption) for a, b in zip(original, reconstructed))
    sheet = Image.new("RGB", (panels[0].width, sum(p.height for p in panels)), "white")
    for row, img in enumerate(panels):
        sheet.paste(img, (0, row*img.height))
    sheet.save(output / "reconstructions.png")
    bad_mse, bad_index, (original, reconstructed) = worst
    panel(original[8], reconstructed[8],
          f"Worst sampled clip {bad_index} | PSNR {psnr(bad_mse):.2f} dB").save(output / "worst_clip.png")
    animation[0].save(output / "reconstructions.gif", save_all=True, append_images=animation[1:],
                      duration=100, loop=0)
    frames_dir = output / "preview_frames"
    frames_dir.mkdir(exist_ok=True)
    for index, img in enumerate(animation):
        img.save(frames_dir / f"{index:04d}.png")
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-framerate", "10",
                    "-i", str(frames_dir / "%04d.png"), "-c:v", "libx264", "-crf", "16",
                    "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(output / "reconstructions.mp4")], check=True)
    print(json.dumps({k: v for k, v in metrics.items() if k not in ("code_counts", "per_clip")}, indent=2), flush=True)


if __name__ == "__main__":
    main()
