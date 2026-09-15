"""Fixed-clip validation and visible collapse checks for tokenizer training."""
import json
import math
from pathlib import Path

import torch
from PIL import Image, ImageDraw


def collapse_reasons(metrics):
    reasons = []
    if metrics["active_codes"] < 2:
        reasons.append("all validation tokens use one code")
    if metrics["output_difference"] < 1e-6:
        reasons.append("different validation clips produce identical outputs")
    if metrics["saturated_fraction"] > 0.99:
        reasons.append("more than 99% of output pixels are saturated")
    return reasons


@torch.no_grad()
def initialize_codebook(model, dataset, device, count=8):
    """Sample feature vectors from several videos, before DDP initialization."""
    features = []
    positions = torch.linspace(0, len(dataset)-1, min(count, len(dataset))).long().tolist()
    training = model.training
    model.eval()
    try:
        for index in positions:
            video = dataset[index][None].to(device)
            with torch.autocast(device.type, dtype=torch.bfloat16,
                                enabled=device.type == "cuda" and torch.cuda.is_bf16_supported()):
                vectors = model.encode_features(video).float().reshape(-1, model.cb.code_width)
            # Bound memory while sampling positions across every frame.
            take = torch.linspace(0, len(vectors)-1, model.cb.K, device=device).long()
            features.append(vectors[take])
        model.cb.initialize_from_data(torch.cat(features))
    finally:
        model.train(training)


@torch.inference_mode()
def evaluate(model, clips, device, *, step, output_dir=None):
    if len(clips) < 2:
        raise ValueError("Validation requires at least two distinct clips")
    training = model.training
    model.eval()
    counts = torch.zeros(model.cb.K, dtype=torch.long)
    mse, baseline_mse, saturation = [], [], []
    predictions, previews = [], []
    use_amp = device.type == "cuda"
    dtype = torch.bfloat16 if use_amp and torch.cuda.is_bf16_supported() else torch.float16
    try:
        for clip in clips:
            video = clip[None].to(device)
            with torch.autocast(device.type, dtype=dtype, enabled=use_amp):
                result = model(video, return_details=True)
            prediction = result["reconstruction"].float()
            counts += torch.bincount(result["indices"].cpu().flatten(), minlength=model.cb.K)
            mse.append((prediction-video).square().mean().item())
            baseline = video.mean(dim=(1,2,3), keepdim=True)
            baseline_mse.append((video-baseline).square().mean().item())
            saturation.append(((prediction < 1e-4) | (prediction > 1-1e-4)).float().mean().item())
            predictions.append(prediction.cpu())
            previews.append((clip[0], prediction[0,0].cpu()))
    finally:
        model.train(training)
    probabilities = counts.double() / counts.sum()
    nonzero = probabilities[probabilities > 0]
    average = sum(mse)/len(mse)
    metrics = dict(step=step, mse=average, psnr=-10*math.log10(max(average, 1e-12)),
        mean_color_baseline_mse=sum(baseline_mse)/len(baseline_mse),
        active_codes=int((counts>0).sum()), total_codes=model.cb.K,
        perplexity=float(torch.exp(-(nonzero*nonzero.log()).sum())),
        dominant_code_fraction=float(probabilities.max()),
        saturated_fraction=sum(saturation)/len(saturation),
        output_difference=max((p-predictions[0]).abs().max().item() for p in predictions),
        per_clip_mse=mse)
    metrics["collapse_reasons"] = collapse_reasons(metrics)
    if output_dir is not None:
        output = Path(output_dir); output.mkdir(parents=True, exist_ok=True)
        with (output/"validation.jsonl").open("a") as handle:
            handle.write(json.dumps(metrics)+"\n")
        height, width = clips[0].shape[1:3]
        scale = 2
        canvas = Image.new("RGB", (width*scale*2, (height*scale+20)*len(previews)+30), "white")
        draw = ImageDraw.Draw(canvas)
        draw.text((8,8), f"Step {step}: input (left) | reconstruction (right)", fill="black")
        for row, pair in enumerate(previews):
            for col, frame in enumerate(pair):
                pixels = (frame.clamp(0,1)*255).byte().numpy()
                img = Image.fromarray(pixels).resize((width*scale,height*scale))
                canvas.paste(img,(col*width*scale,30+row*(height*scale+20)))
        canvas.save(output/f"step_{step:07d}.png")
    return metrics
