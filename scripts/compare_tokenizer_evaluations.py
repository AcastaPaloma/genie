"""Compare identical held-out samples and render original/earlier/latest video panels."""
import argparse
import json
from pathlib import Path
import statistics
import subprocess

from PIL import Image, ImageDraw


def load(directory):
    return json.loads((directory / "metrics.json").read_text())


def matching_samples(reference, candidate):
    for key in ("seed", "clips", "frames", "content_size", "padded_size", "fixed_preview_indices"):
        if reference[key] != candidate[key]:
            raise ValueError(f"Evaluations differ in {key}")
    a = [(r["index"], r["record_and_start"]) for r in reference["per_clip"]]
    b = [(r["index"], r["record_and_start"]) for r in candidate["per_clip"]]
    if a != b:
        raise ValueError("Sampled video windows differ")


def comparison(original, earlier, latest, caption, old_step, new_step):
    width, height = original.size
    canvas = Image.new("RGB", (width * 3, height + 44), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 5), caption, fill="black")
    for column, (label, frame) in enumerate(zip(
            ("Original", f"Earlier: step {old_step}", f"Latest: step {new_step}"),
            (original, earlier, latest))):
        draw.text((column * width + 8, 24), label, fill="black")
        canvas.paste(frame, (column * width, 44))
    return canvas


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--earlier", type=Path, required=True)
    parser.add_argument("--latest", type=Path, required=True)
    parser.add_argument("--best", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    before, after = load(args.earlier), load(args.latest)
    matching_samples(before, after)
    best = load(args.best) if args.best else None
    if best:
        matching_samples(before, best)
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    old_step = before["checkpoint_metadata"]["global_step"]
    new_step = after["checkpoint_metadata"]["global_step"]
    old_scores = {r["index"]: r["psnr"] for r in before["per_clip"]}
    new_scores = {r["index"]: r["psnr"] for r in after["per_clip"]}
    frames = []
    length = before["frames"] // before["clips"]
    for position in range(len(before["fixed_preview_indices"]) * length):
        index = before["fixed_preview_indices"][position // length]
        previous = Image.open(args.earlier / "preview_frames" / f"{position:04d}.png").convert("RGB")
        current = Image.open(args.latest / "preview_frames" / f"{position:04d}.png").convert("RGB")
        if previous.size != current.size:
            raise ValueError("Preview sizes differ")
        width = previous.width // 2
        box = (0, 44, width, previous.height)
        original = previous.crop(box)
        if original.tobytes() != current.crop(box).tobytes():
            raise ValueError(f"Original preview frames differ at frame {position}")
        box = (width, 44, width * 2, previous.height)
        caption = f"Test clip {index} | clip PSNR: {old_scores[index]:.2f} -> {new_scores[index]:.2f} dB"
        frames.append(comparison(original, previous.crop(box), current.crop(box),
                                 caption, old_step, new_step))
    selected = [frames[i * length + min(8, length-1)] for i in range(len(before["fixed_preview_indices"]))]
    sheet = Image.new("RGB", (selected[0].width, sum(p.height for p in selected)), "white")
    for row, panel in enumerate(selected):
        sheet.paste(panel, (0, row * panel.height))
    sheet.save(output / "comparison.png")
    frames[0].save(output / "comparison.gif", save_all=True, append_images=frames[1:],
                   duration=100, loop=0)
    frame_dir = output / "comparison_frames"
    frame_dir.mkdir(exist_ok=True)
    for index, frame in enumerate(frames):
        frame.save(frame_dir / f"{index:04d}.png")
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-framerate", "10",
                    "-i", str(frame_dir / "%04d.png"), "-c:v", "libx264", "-crf", "16",
                    "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(output / "comparison.mp4")], check=True)
    fields = ("checkpoint_metadata", "content_mse", "content_psnr", "content_mae", "active_codes",
              "code_perplexity", "dominant_code_fraction", "saturated_fraction", "worst_clip_index",
              "per_clip_psnr_percentiles", "fixed_validation")
    summary = {"earlier": {k: before[k] for k in fields}, "latest": {k: after[k] for k in fields},
               "clips": after["clips"], "frames": after["frames"], "seed": after["seed"],
               "mse_reduction_percent": 100 * (1-after["content_mse"]/before["content_mse"]),
               "psnr_gain_db": after["content_psnr"]-before["content_psnr"],
               "clips_improved": sum(new_scores[i] > old_scores[i] for i in old_scores)}
    if best:
        summary["saved_best"] = {k: best[k] for k in fields}
        summary["latest_mse_reduction_vs_saved_best_percent"] = 100 * (1-after["content_mse"]/best["content_mse"])
    history = output / "training_validation.jsonl"
    if history.exists():
        rows = [json.loads(line) for line in history.read_text().splitlines() if line.strip()]
        rows = [r for r in rows if r["step"] <= new_step]
        old, new = rows[-24:-12], rows[-12:]
        a, b = (statistics.mean(r["mse"] for r in group) for group in (old, new))
        summary["fixed_validation_recent_trend"] = dict(
            earlier_steps=[old[0]["step"], old[-1]["step"]], latest_steps=[new[0]["step"], new[-1]["step"]],
            earlier_mean_mse=a, latest_mean_mse=b, reduction_percent=100*(1-b/a))
    (output / "comparison.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
