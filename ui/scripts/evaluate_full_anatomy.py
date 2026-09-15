"""Optimistic, bounded whole-neuron brightness experiment on complete anatomy.

This is a feasibility diagnostic, never a neural output for the application.
The convex first-order gap provides an upper bound on further improvement of
this particular fixed-camera, transparent, cable-length observation model.
"""
import argparse
import json
import subprocess
import time

import numpy as np
import imageio_ffmpeg
from scipy.sparse import load_npz
from PIL import Image, ImageDraw

from prepare_full_brain import OUT, SOURCE


def fit(basis, target, iterations):
    mask = np.diff(basis.indptr) > 0
    h = basis[mask].astype(np.float64)
    ht = h.T.tocsr()
    y = target.ravel()[mask]
    scale = np.maximum(np.asarray(ht.sum(axis=1)).ravel(), 1e-15)
    x = np.full(h.shape[1], y.mean()); z = x.copy(); acceleration = 1.
    started = time.monotonic(); best = np.inf; best_x = x.copy()
    records = []
    for iteration in range(iterations):
        residual = h @ z - y
        next_x = np.clip(z - (ht @ residual) / scale, 0, 1)
        next_acceleration = (1 + np.sqrt(1 + 4 * acceleration**2)) / 2
        z = next_x + (acceleration - 1) / next_acceleration * (next_x - x)
        x = next_x; acceleration = next_acceleration
        if iteration % 50 == 0 or iteration == iterations - 1:
            r = h @ x - y; mse = float(r @ r / len(y))
            if mse < best: best = mse; best_x = x.copy()
            gradient = 2 * (ht @ r) / len(y)
            gap = float(np.maximum(gradient, 0) @ x + np.minimum(gradient, 0) @ (x - 1))
            records.append(dict(iteration=iteration + 1, mse=mse, improvementUpperBound=max(0., gap)))
            print(f"iteration {iteration+1}: MSE {mse:.7f}, remaining improvement <= {gap:.7f}, {time.monotonic()-started:.1f}s", flush=True)
            if gap < 1e-6: break
    r = h @ best_x - y; gradient = 2 * (ht @ r) / len(y)
    gap = max(0., float(np.maximum(gradient, 0) @ best_x + np.minimum(gradient, 0) @ (best_x - 1)))
    return (basis @ best_x).reshape(target.shape), dict(mse=best, psnr=float(-10*np.log10(max(best, 1e-20))),
        constantMse=float(np.var(y)), lowerBoundMse=max(0., best-gap), improvementUpperBound=gap,
        seconds=time.monotonic()-started, progress=records)


def main(iterations):
    meta = json.loads((OUT / 'whole-arbor-320.json').read_text())
    width, height = meta['resolution']; basis = load_npz(SOURCE / 'whole-arbor-320.npz')
    yy, xx = np.indices((height, width))
    targets = [('Checkerboard · 16-pixel squares', ((xx//16+yy//16) % 2).astype(float))]
    clips = json.loads((OUT.parent / 'live-clips/clips.json').read_text())['clips']
    for clip in clips[:3]:
        raw = subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(),'-v','error','-ss','2','-i',str(OUT.parent / clip['file']),'-frames:v','1',
                              '-vf',f'scale={width}:{height},format=rgb24','-f','rawvideo','pipe:1'],capture_output=True,check=True).stdout
        rgb = np.frombuffer(raw, np.uint8).reshape(height,width,3).astype(float)/255
        targets.append((clip['name'], rgb @ np.array([.2126,.7152,.0722])))
    panel = Image.new('RGB', (width*2+36, (height+58)*len(targets)+55), '#f6f6f2'); draw = ImageDraw.Draw(panel)
    draw.text((12, 8), 'FULL ANATOMY · independent whole-neuron brightness fit · NOT simulated spikes', fill='#263238')
    draw.text((12, 29), 'Source target', fill='#263238'); draw.text((width+24, 29), 'Best measured fit to the complete cable projection', fill='#263238')
    records=[]
    for index, (label, target) in enumerate(targets):
        print(label, flush=True); prediction, metrics = fit(basis,target,iterations)
        records.append(dict(label=label,**metrics))
        y = 55+index*(height+58)
        panel.paste(Image.fromarray(np.uint8(np.clip(target,0,1)*255)).convert('RGB'),(12,y))
        panel.paste(Image.fromarray(np.uint8(np.clip(prediction,0,1)*255)).convert('RGB'),(width+24,y))
        draw.text((12,y+height+8), f'{label} | PSNR {metrics["psnr"]:.2f} dB | MSE {metrics["mse"]:.5f}', fill='#263238')
        draw.text((12,y+height+25), f'Certified MSE lower bound for this linear model: {metrics["lowerBoundMse"]:.5f}', fill='#263238')
        panel.save(OUT/'independent-brightness-comparison.png')
        (OUT/'independent-brightness-results.json').write_text(json.dumps(dict(
            status='anatomical feasibility experiment, not neural playback', anatomy=meta,
            method='Bounded convex whole-neuron brightness fit with first-order optimality gap. Every released cable contributes.',
            limits='This fixed transparent observation model does not prove impossibility for other optics, cameras or multicompartment models.',
            results=records),indent=2)+'\n')


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--iterations',type=int,default=1000)
    main(parser.parse_args().iterations)
