"""Review an actual complete-brain spike sequence on frozen, unseen inputs.

The exported movie is a recorded diagnostic, never used by the application to
play neural activity. Runtime still computes spikes from the supplied video.
"""
import argparse
import base64
import hashlib
import json
import subprocess
from pathlib import Path

import imageio_ffmpeg
import numpy as np
import scipy.ndimage as ndi
import scipy.sparse as sp
from PIL import Image, ImageDraw, ImageFont

root = Path(__file__).resolve().parents[1]
records = root / ".cache/full-brain/validation"
parser = argparse.ArgumentParser()
parser.add_argument('--label', default='event-frozen-unseen-v1')
parser.add_argument('--source-name', default='unseen-maps-640')
parser.add_argument('--movie-name', default='recorded-spike-comparison')
args = parser.parse_args()
if any(not name.replace('-', '').replace('_', '').isalnum() for name in [args.label, args.source_name, args.movie_name]):
    raise ValueError('Use simple artifact names')
label = args.label
summary = json.loads((records / f"{label}-summary.json").read_text())["summary"]
provenance = json.loads((root / f"public/data/live-evaluation/{args.source_name}.json").read_text())
if summary['sourceUrl'] != '/data/live-evaluation/' + provenance['file']:
    raise ValueError('The evaluated input differs from the frozen source')
if hashlib.sha256((root / 'public/data/live-evaluation' / provenance['file']).read_bytes()).hexdigest() != provenance['sha256']:
    raise ValueError('The frozen input file changed')
if summary["status"] != "completed" or summary["parameters"] != provenance["frozenController"]:
    raise ValueError("The completed controller differs from the frozen evaluation parameters")
width = summary.get('controlWidth', 320)
if width not in [320, 640] or width != provenance.get('controlWidth', 320):
    raise ValueError('The control resolution differs from the frozen evaluation')
height = width * 3 // 4
for name, digest in provenance["controllerSourceSha256"].items():
    if hashlib.sha256((root / "src" / name).read_bytes()).hexdigest() != digest:
        raise ValueError("Controller source changed since the input was generated")
    if provenance.get('controllerSnapshotDirectory'):
        archived = root / 'public/data/live-evaluation' / provenance['controllerSnapshotDirectory'] / name
        if hashlib.sha256(archived.read_bytes()).hexdigest() != digest:
            raise ValueError('Archived frozen controller source differs')
items = [json.loads((records / f"{label}-f{i:04d}-map01.json").read_text()) for i in range(120)]
if any(abs(item["metrics"]["time"] - i / 10) > 1e-6 for i, item in enumerate(items)):
    raise ValueError("Sequence timestamps differ")
def pixels(item):
    if item.get('resolution', [320, 240]) != [width, height]:
        raise ValueError('Recorded pixel dimensions differ')
    if 'pixelsBase64' in item:
        packed = np.frombuffer(base64.b64decode(item['pixelsBase64'], validate=True), '<f4')
    else:
        packed = np.asarray(item['target'] + item['prediction'], dtype=np.float32)
    if packed.shape != (width * height * 2,) or not np.isfinite(packed).all():
        raise ValueError('Incomplete or nonfinite recorded pixels')
    return packed.reshape(2, height, width)

decoded = np.asarray([pixels(item) for item in items])
targets, observed = decoded[:, 0], decoded[:, 1]
h = sp.load_npz(root / f".cache/full-brain/whole-arbor-{width}.npz")
support = (np.asarray(h.sum(axis=1)).ravel() > 0).reshape(height, width)
interior = ndi.binary_erosion(support, iterations=2)
if np.any(observed[:, ~support] != 0):
    raise ValueError("Observed light exists outside the complete-arbor operator's support")
comparisons = []
for segment in provenance["segments"]:
    start = int(segment["startSeconds"] * 10)
    y, p = targets[start:start + 40], observed[start:start + 40]
    a, b = y[:, support], p[:, support]
    a0, b0 = a - a.mean(axis=1, keepdims=True), b - b.mean(axis=1, keepdims=True)
    correlation = np.sum(a0 * b0, axis=1) / np.maximum(1e-10, np.sqrt(np.sum(a0 * a0, axis=1) * np.sum(b0 * b0, axis=1)))
    sy = np.stack([ndi.sobel(y, axis=axis)[:, interior] / 8 for axis in [1, 2]])
    sp_ = np.stack([ndi.sobel(p, axis=axis)[:, interior] / 8 for axis in [1, 2]])
    motion_y, motion_p = np.diff(a, axis=0), np.diff(b, axis=0)
    # Identical interior time windows for every lag; never compare across cuts.
    times = np.arange(4, 36)
    lags = {str(int(lag * 100)): float(np.mean((b[times] - a[times - lag]) ** 2)) for lag in range(-3, 4)}
    comparisons.append({
        "map": segment["map"], "startSeconds": segment["startSeconds"],
        "supportedMse": float(np.mean((a - b) ** 2)), "fullFrameMse": float(np.mean((y - p) ** 2)),
        "constantImageMse": float(np.mean(a0 * a0)), "meanImageCorrelation": float(correlation.mean()),
        "interiorEdgeCosine": float(np.sum(sy * sp_) / max(1e-10, np.linalg.norm(sy) * np.linalg.norm(sp_))),
        "interiorEdgeAmplitudeRatio": float(np.linalg.norm(sp_) / max(1e-10, np.linalg.norm(sy))),
        "motionDifferenceMse": float(np.mean((motion_y - motion_p) ** 2)),
        "noMotionDifferenceMse": float(np.mean(motion_y * motion_y)),
        "lagMseByMilliseconds": lags, "lowestTestedLagMs": int(min(lags, key=lags.get)),
    })
movie = root / f"public/data/live-evaluation/{args.movie_name}.mp4"
movie_width, movie_height = width * 2 + 32, height + 72
encoder = subprocess.Popen([
    imageio_ffmpeg.get_ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-y",
    "-f", "rawvideo", "-pixel_format", "rgb24", "-video_size", f"{movie_width}x{movie_height}",
    "-framerate", "10", "-i", "-", "-an", "-c:v", "libx264", "-crf", "16",
    "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(movie)
], stdin=subprocess.PIPE)
font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 13)
for i, (y, p) in enumerate(zip(targets, observed)):
    frame = Image.new("RGB", (movie_width, movie_height), "#101619")
    draw = ImageDraw.Draw(frame)
    draw.text((8, 8), "Source supplied to controller", fill="white", font=font)
    draw.text((width + 24, 8), "Recorded actual spikes · fixed 1.5× exposure", fill="white", font=font)
    for x, values in [(8, y), (width + 24, p * 1.5)]:
        frame.paste(Image.fromarray(np.uint8(np.clip(values, 0, 1) * 255)).convert("RGB"), (x, 34))
    caption = f"{i/10:.1f} s · 139,255 neurons · all branches in the {width} × {height} operator"
    if summary['parameters'].get('useVisualInputs'):
        direct = len(summary['visualInputIds'])
        seeds = f"{summary['seedCount']:,} seeds" if summary['seedCount'] else 'no constant seeds'
        caption = f"{i/10:.1f} s · All arbors · {direct:,} direct visual inputs; {seeds}"
    draw.text((8, height + 43), caption, fill="#bec8ce", font=font)
    encoder.stdin.write(frame.tobytes())
encoder.stdin.close()
if encoder.wait():
    raise RuntimeError("Recording encoding failed")
result = {
    "scope": f"Frozen controller, three previously unused maps, two cuts, persistent neural history. Numerical {width}x{height} complete-arbor operator; excludes full UI/body timing.",
    "controlWidth": width,
    "sourceSha256": provenance["sha256"], "inputMechanism": summary.get("inputMechanism", "Video URL"), "controller": summary["parameters"], "frames": 120,
    "distinctDecodedTargets": len({hashlib.sha256(x.tobytes()).hexdigest() for x in targets}),
    "distinctObservedFrames": len({hashlib.sha256(x.tobytes()).hexdigest() for x in observed}),
    "meanNeuralMs": summary["meanComputeMs"], "cut": summary["cut"],
    "perScene": comparisons, "recordedDiagnostic": str(movie),
    "interpretation": "Metrics supplement visual review; they do not establish recognizable detail. The recording contains measured neural output and is never a runtime fallback. If these inputs guide tuning, a new holdout is required."
}
if all('visualInputLightFraction' in frame for frame in summary['frames']):
    result['lightAttribution'] = {'scope': f'Complete {width} × {height} control projection; optical contribution, not information origin',
        'meanVisualInputFraction': float(np.mean([f['visualInputLightFraction'] for f in summary['frames']])),
        'meanSeedFraction': float(np.mean([f['seedLightFraction'] for f in summary['frames']])),
        'meanSynapseDependentFraction': float(np.mean([f['synapseDependentLightFraction'] for f in summary['frames']]))}
(records / f"{label}-review.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result, indent=2))
