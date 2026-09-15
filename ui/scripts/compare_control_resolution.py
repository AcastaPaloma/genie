"""Compare stored actual spike traces at one shared anatomical resolution.

Changing control resolution changes the sampling/support of its training loss.
This independent comparison reprojects every cell through the same full-cable
operator, so lower loss cannot result merely from changing the reported pixels.
It is an offline measurement, never a source of runtime activity.
"""
import argparse
import base64
import hashlib
import json
from pathlib import Path

import numpy as np
import scipy.ndimage as ndi
import scipy.sparse as sp
from PIL import Image, ImageDraw, ImageFont

root = Path(__file__).resolve().parents[1]
records = root / ".cache/full-brain/validation"
parser = argparse.ArgumentParser()
parser.add_argument("--observation-width", type=int, choices=[320, 640, 1280], default=320)
parser.add_argument("--labels", nargs=2, default=["event-support-image-guard", "event-640-guarded-v1"])
parser.add_argument("--snapshots", nargs="+", type=int)
parser.add_argument("--source-name")
parser.add_argument("--name", default="control-resolution-comparison")
args = parser.parse_args()
width = args.observation_width
height = width * 3 // 4
labels = args.labels
summaries = [json.loads((records / f"{label}-summary.json").read_text())["summary"] for label in labels]
source_meta = None
if args.snapshots:
    if not args.source_name:
        parser.error("Sequence snapshots require --source-name to verify the common native input")
    source_dir = root / "public/data/live-evaluation"
    source_meta = json.loads((source_dir / f"{args.source_name}.json").read_text())
    if hashlib.sha256((source_dir / source_meta["file"]).read_bytes()).hexdigest() != source_meta["sha256"]:
        raise ValueError("Source fixture hash differs")
    for summary in summaries:
        if summary.get("sourceUrl") != f"/data/live-evaluation/{source_meta['file']}":
            raise ValueError("The controllers did not process the same verified input")
        if summary.get("sourceNativeResolution") != source_meta["nativeResolution"]:
            raise ValueError("Decoded input resolution differs from the fixture")
    if summaries[0]["parameters"] != summaries[1]["parameters"]:
        raise ValueError("Controller parameters differ beyond resolution")
    if summaries[0].get("sourceStart") != summaries[1].get("sourceStart"):
        raise ValueError("Sequence histories start at different source times")
    if [f["time"] for f in summaries[0]["frames"]] != [f["time"] for f in summaries[1]["frames"]]:
        raise ValueError("Sequence histories differ")
if summaries[0]["anatomySha256"] != summaries[1]["anatomySha256"]:
    raise ValueError("The controllers used different anatomy")
if summaries[0].get("aperture") != summaries[1].get("aperture"):
    raise ValueError("Source apertures differ")
observation_meta = json.loads((root / f"public/data/full-brain-783/whole-arbor-{width}.json").read_text())
if observation_meta["sourceAnatomySha256"] != summaries[0]["anatomySha256"]:
    raise ValueError("The observation anatomy differs from the simulated anatomy")
h = sp.load_npz(root / f".cache/full-brain/whole-arbor-{width}.npz")
if h.shape != (width * height, 139255):
    raise ValueError("Incomplete observation operator")
mask = (np.asarray(h.sum(axis=1)).ravel() > 0).reshape(height, width)
interior = ndi.binary_erosion(mask, iterations=2)
samples = [("map01", index) for index in args.snapshots] if args.snapshots else [(clip, None) for clip in ["map01", "map02", "map03"]]
panel = Image.new("RGB", (1488, 112 + len(samples) * 398), "#101619")
draw = ImageDraw.Draw(panel)
font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 17)
small = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 14)
draw.text((16, 12), "Same neurons and anatomy: does finer control preserve more detail?", font=font, fill="white")
draw.text((16, 40), f"Actual simulated spikes; both outputs projected through the complete {width} × {height} operator. Fixed 1.5× exposure.", font=small, fill="#cad3d8")
for i, title in enumerate(["Source", "320 × 240 control", "640 × 480 control"]):
    draw.text((16 + i * 488, 69), title, font=font, fill="white")
comparisons = []
target_resolutions = []
for row, (clip, snapshot) in enumerate(samples):
    suffix = f"-f{snapshot:04d}" if snapshot is not None else ""
    items = [json.loads((records / f"{label}{suffix}-{clip}.json").read_text()) for label in labels]
    # Native-sequence trials use the finer decoded target, never a magnified
    # coarse target. Legacy mode preserves its historical 320-pixel reference.
    target_record = items[int(np.argmax([np.prod(item.get("resolution", [320, 240])) for item in items]))] if args.snapshots else items[0]
    target_width, target_height = target_record.get("resolution", [320, 240])
    if "pixelsBase64" in target_record:
        packed = np.frombuffer(base64.b64decode(target_record["pixelsBase64"], validate=True), "<f4")
        if packed.shape != (2 * target_width * target_height,):
            raise ValueError("Incomplete recorded pixels")
        target = packed[:target_width * target_height].reshape(target_height, target_width)
    else:
        target = np.asarray(target_record["target"], dtype=np.float32).reshape(target_height, target_width)
    target_resolutions.append([target_width, target_height])
    if (target_width, target_height) != (width, height):
        target = np.asarray(Image.fromarray(target).resize((width, height), Image.Resampling.BILINEAR))
    expected_time = target_record["metrics"]["time"]
    scene = clip
    if source_meta:
        scene = [s["map"] for s in source_meta["segments"] if s["startSeconds"] <= expected_time][-1]
    outputs = []
    for label, item, summary in zip(labels, items, summaries):
        if item["metrics"]["time"] != expected_time:
            raise ValueError("Source times differ")
        if args.snapshots and item.get("sourceUrl") != summary["sourceUrl"]:
            raise ValueError("Snapshot and summary inputs differ")
        rates = np.frombuffer(base64.b64decode(item["neuronRatesBase64"], validate=True), "<f4")
        if rates.shape != (139255 * 2,) or summary["neurons"] != 139255 or not np.isfinite(rates).all():
            raise ValueError("Incomplete recorded neural state")
        actual = np.clip(rates[139255:] / summary["parameters"]["rateScale"], 0, 1)
        predicted = (h @ actual).reshape(height, width)
        if np.any(predicted[~mask] != 0):
            raise ValueError("Light outside anatomical support")
        a = np.stack([ndi.sobel(target, axis=axis)[interior] / 8 for axis in [0, 1]])
        b = np.stack([ndi.sobel(predicted, axis=axis)[interior] / 8 for axis in [0, 1]])
        metrics = dict(label=label, clip=scene, time=expected_time,
                       supportedMse=float(np.mean((predicted[mask] - target[mask]) ** 2)),
                       fullFrameMse=float(np.mean((predicted - target) ** 2)),
                       interiorEdgeCosine=float(np.sum(a * b) / max(1e-10, np.linalg.norm(a) * np.linalg.norm(b))))
        outputs.append(predicted)
        comparisons.append(metrics)
    y = 98 + row * 398
    for i, values in enumerate([target] + [p * 1.5 for p in outputs]):
        raster = Image.fromarray(np.uint8(np.clip(values, 0, 1) * 255)).convert("RGB")
        panel.paste(raster.resize((480, 360), Image.Resampling.BOX if width > 480 else Image.Resampling.NEAREST), (16 + i * 488, y))
    draw.text((16, y + 364), f"{scene} · {expected_time:.1f} s", font=small, fill="#cad3d8")
    for col, metric in enumerate(comparisons[-2:], 1):
        draw.text((16 + col * 488, y + 364), f"MSE {metric['supportedMse']:.5f} · edge similarity {metric['interiorEdgeCosine']:.3f}", font=small, fill="#cad3d8")
result = dict(scope=f"{len(samples)} tuning frames, common complete {width} × {height} observation operator; not held-out recognition proof",
              targetSampling="Finer controller's browser-decoded luminance" if args.snapshots else "Recorded 320 × 240 decoded luminance",
              targetResolutions=target_resolutions, resizing="Bilinear resizing to observation dimensions, when different; enlargement adds no source detail",
              sourceSha256=source_meta["sha256"] if source_meta else None, anatomySha256=observation_meta["sourceAnatomySha256"],
              observationOperatorSha256=observation_meta["operatorSha256"], comparisons=comparisons)
for label, summary in zip(labels, summaries):
    result[label] = dict(meanNeuralMs=summary["meanComputeMs"], meanNativeSupportedMse=float(np.mean([f["maskedMse"] for f in summary["frames"]])), nativeResolution=summary.get("resolution", [320, 240]), cut=summary["cut"], darkFrames=summary["darkFrames"])
suffix = f"-{width}" if width != 320 else ""
panel.save(records / f"{args.name}{suffix}.png")
(records / f"{args.name}{suffix}.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result, indent=2))
