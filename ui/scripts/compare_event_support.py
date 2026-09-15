"""Matched before/after measurements from actual complete-brain spike output."""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import binary_erosion, sobel
from scipy.sparse import load_npz

root = Path(__file__).resolve().parents[1]
records = root / '.cache/full-brain/validation'
parser = argparse.ArgumentParser()
modes=parser.add_mutually_exclusive_group()
modes.add_argument('--visual-inputs', action='store_true')
modes.add_argument('--weak-inputs', action='store_true')
args = parser.parse_args()
labels = ['event-visual-dense-v1', 'event-visual-weak-input-v1'] if args.weak_inputs else ['event-support-precomputed-v1', 'event-visual-dense-v1'] if args.visual_inputs else ['event-observed-image-v1', 'event-support-precomputed-v1']
name = 'weak-input-comparison' if args.weak_inputs else 'visual-input-comparison' if args.visual_inputs else 'event-support-comparison'
h = load_npz(root / '.cache/full-brain/whole-arbor-320.npz')
mask = (np.asarray(h.sum(axis=1)).ravel() > 0).reshape(240, 320)
inside = binary_erosion(mask, iterations=2)
panel = Image.new('RGB', (1024, 990), '#101619')
draw = ImageDraw.Draw(panel)
font = ImageFont.truetype('/System/Library/Fonts/Helvetica.ttc', 15)
small = ImageFont.truetype('/System/Library/Fonts/Helvetica.ttc', 12)
draw.text((16, 14), 'Full brain: using weaker synaptic inputs, without constant seed drive' if args.weak_inputs else 'Full brain: visual input stimulation and denser spikes' if args.visual_inputs else 'Actual full-brain spikes: feedback through signed connections', font=font, fill='white')
draw.text((16, 40), '11,391 direct visual inputs. Every arbor visible; fixed 1.5× exposure. New preparation stays below the unchanged threshold.' if args.weak_inputs else 'Trial: 11,391 directly stimulated visual inputs + 700 seeds. Every arbor visible; 1.5× exposure on both outputs.' if args.visual_inputs else '139,255 neurons · every released branch · fixed 1.5× exposure on both activity columns', font=small, fill='#c4cfd6')
for i, title in enumerate(['Source frame', 'Previous visual-input trial', 'Updated visual-input trial'] if args.weak_inputs else ['Source frame', 'Earlier seed controller', 'Earlier visual-input trial'] if args.visual_inputs else ['Source frame', 'Previous image feedback', 'Current signed feedback']):
    draw.text((16 + i * 336, 70), title, font=font, fill='white')
metrics = []
for row, clip in enumerate(['map01', 'map02', 'map03']):
    items = [json.loads((records / f'{label}-{clip}.json').read_text()) for label in labels]
    target = np.asarray(items[0]['target'], dtype=np.float32).reshape(240, 320)
    if not np.allclose(target.ravel(), items[1]['target'], atol=1e-6, rtol=0):
        raise ValueError('Matched source frames differ')
    predicted = [np.asarray(item['prediction'], dtype=np.float32).reshape(240, 320) for item in items]
    y = 100 + row * 290
    for col, value in enumerate([target] + [p * 1.5 for p in predicted]):
        panel.paste(Image.fromarray(np.uint8(np.clip(value, 0, 1) * 255)).convert('RGB'), (16 + col * 336, y))
    draw.text((16, y + 245), f'{clip} · source {items[0]["metrics"]["time"]:.1f} s', font=small, fill='#c4cfd6')
    a = np.stack([sobel(target, axis=axis)[inside] for axis in [0, 1]])
    for col, (label, item, prediction) in enumerate(zip(labels, items, predicted), 1):
        b = np.stack([sobel(prediction, axis=axis)[inside] for axis in [0, 1]])
        edge = float(np.sum(a * b) / max(1e-10, np.linalg.norm(a) * np.linalg.norm(b)))
        metrics.append(dict(label=label, clip=clip, supportedMse=item['metrics']['maskedMse'], interiorEdgeCosine=edge))
        draw.text((16 + col * 336, y + 245), f'MSE {item["metrics"]["maskedMse"]:.5f} · edge similarity {edge:.3f}', font=small, fill='#c4cfd6')
draw.text((16, 971), 'Tuning frames, not a recognition test. The current model still loses fine detail. No source pixels are composited into brain output.', font=small, fill='#c4cfd6')
panel.save(root / f'public/data/full-brain-783/{name}.png')
result = dict(scope='Three matched tuning frames, complete 320 × 240 cable operator', frames=metrics)
for label in labels:
    summary = json.loads((records / f'{label}-summary.json').read_text())['summary']
    result[label] = dict(meanSupportedMse=float(np.mean([f['maskedMse'] for f in summary['frames']])), meanNeuralMs=summary['meanComputeMs'], cut=summary['cut'])
(records / f'{name}.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result, indent=2))
