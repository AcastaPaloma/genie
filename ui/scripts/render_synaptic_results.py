"""Publish the measured sequence results, comparison, and recorded playback.

Images come from actual simulated output spike traces projected through the
anatomical operator. Desired neuron activities are never substituted.
"""
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT/'public/data'
CACHE = ROOT/'.cache/control-goal'
runs = json.loads((CACHE/'sequence-validation.json').read_text())
ablation = json.loads((CACHE/'sequence-ablation.json').read_text())
operator = json.loads((DATA/'synaptic-image-operator.json').read_text())
controller = json.loads((DATA/'presynaptic-controller.json').read_text())
summary = []
for run in runs:
    frames = run['frames']
    mse = float(np.mean([f['mse'] for f in frames]))
    flat = float(np.mean([f['flatMse'] for f in frames]))
    summary.append(dict(clip=run['clip'], frames=len(frames),
                        medianPsnr=float(np.median([f['psnr'] for f in frames])),
                        minimumPsnr=min(f['psnr'] for f in frames), meanMse=mse,
                        meanFlatMse=flat, relativeMseReduction=1-mse/flat,
                        wallSeconds=run['wallMs']/1000))
report = dict(method='Live feedback through the full published graph. Only measured output spike traces reach the renderer. No video textures or direct stimulation of displayed or motor-output neurons.',
              protocol='Four clips, 16 consecutive frames each; seeded reset and 300 ms warmup per clip, then 100 ms per frame without resetting state. Score all 1200 grayscale pixels. This is engineering validation after controller/view selection, not a held-out biological/generalization study.',
              timing='10 frames per simulated second. GIF plays recorded simulation at that rate, not measured live wall speed. Timings exclude each clip warmup and the physical body.',
              controls=len(controller['inputIds']), maximumOpticalInputHz=4000,
              displayRateScaleHz=60, spikeTraceMs=100,
              opticalCaveat=controller['caveat'],
              connectomeSha256=controller['connectomeSha256'], operator=operator,
              summary=summary, ablation=ablation, runs=runs)
(DATA/'synaptic-control-validation.json').write_text(json.dumps(report, separators=(',', ':')))

font = ImageFont.load_default(size=17)
small = ImageFont.load_default(size=14)
def tile(values):
    pixels = (np.asarray(values).reshape(30, 40)*255).clip(0, 255).astype('uint8')
    return Image.fromarray(pixels).resize((320, 240), Image.Resampling.NEAREST).convert('RGB')

panel = Image.new('RGB', (704, 1260), '#f8f9fa')
draw = ImageDraw.Draw(panel)
draw.text((16, 12), 'DOOM source / actual synaptic reconstruction', font=font, fill='#242c34')
draw.text((16, 38), '40 x 30 grayscale; recorded full-network simulation; 100 ms spike trace.', font=small, fill='#616d79')
movie = []
for row, run in enumerate(runs):
    frame = run['frames'][8]
    top = 74+row*294
    draw.text((16, top), f'{run["clip"]} / frame {frame["index"]+1} (1-based)', font=small, fill='#242c34')
    draw.text((368, top), f'Measured spikes / {frame["psnr"]:.2f} dB', font=small, fill='#242c34')
    panel.paste(tile(frame['target']), (16, top+26))
    panel.paste(tile(frame['observed']), (368, top+26))
    for f in run['frames']:
        image = Image.new('RGB', (704, 310), '#f8f9fa')
        d = ImageDraw.Draw(image)
        d.text((16, 9), f'{run["clip"]} / frame {f["index"]+1} - source', font=font, fill='#242c34')
        d.text((368, 9), 'Actual synaptic reconstruction', font=font, fill='#242c34')
        image.paste(tile(f['target']), (16, 36)); image.paste(tile(f['observed']), (368, 36))
        d.text((16, 287), 'Recorded at 10 FPS in simulation time; live computation is slower.', font=small, fill='#616d79')
        movie.append(image)
panel.save(DATA/'synaptic-control-comparison.png')
movie[0].save(DATA/'synaptic-playback.gif', save_all=True, append_images=movie[1:], duration=100, loop=0, optimize=False)
print(json.dumps(summary, indent=2))
print('Disconnection:', ablation)
