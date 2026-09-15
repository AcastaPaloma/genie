"""Optimistic whole-neuron brightness fit, before any neural-dynamics constraint.

This uses a linear density-normalized projection operator, not the depth-buffered
interactive renderer. A poor fit limits this basis/model, not every possible fly
display. No claim of a universal impossibility result is made.
"""
from pathlib import Path
import json
import numpy as np
from PIL import Image, ImageDraw
from scipy.optimize import minimize

root = Path(__file__).resolve().parents[1]
out = root / "public/data"
points = np.fromfile(out / "brain-points.bin", dtype="<f4").reshape(-1, 4)
W, H = 40, 30
uv = (points[:, :2] - [0, 25]) / [300, 225] + .5
valid = ((uv >= 0) & (uv < 1)).all(axis=1)
xy = (uv[valid] * [W, H]).astype(int)
pixels = (H-1-xy[:, 1])*W + xy[:, 0]
operator = np.zeros((W*H, 1500), dtype=np.float64)
np.add.at(operator, (pixels, points[valid, 3].astype(int)), 1)
density = operator.sum(axis=1)
operator /= np.maximum(density[:, None], 1)
mask = density > 0
sheet = Image.open(out / "doom-0.webp")
frames = [0, 52, 100]
gy,gx=np.indices((H,W))
targets = [Image.fromarray(((gx//4+gy//4)%2*255).astype('uint8'))]
for frame in frames:
    x, y = frame % 16 * 160, frame//16*90
    targets.append(sheet.crop((x+20, y, x+140, y+90)).convert('L').resize((W, H), Image.Resampling.BOX))
records = []
panel = Image.new('RGB', (640, len(targets)*190), '#f8f9fa')
draw = ImageDraw.Draw(panel)
for j, target in enumerate(targets):
    y = np.asarray(target).astype(float).ravel()/255
    a = operator[mask]; b=y[mask]
    def objective(z):
        residual = a@z-b
        return float(residual@residual/len(b)), 2*a.T@residual/len(b)
    fit = minimize(objective, np.full(1500,b.mean()), jac=True, bounds=[(0,1)]*1500,
                   method='L-BFGS-B', options={'maxiter':600, 'ftol':1e-12, 'gtol':1e-7})
    pred=operator@fit.x
    mse=float(np.mean((pred[mask]-b)**2)); baseline=float(np.mean((b-b.mean())**2))
    label='Checker pattern' if j==0 else f'DOOM frame {frames[j-1]+1}'
    records.append(dict(label=label, mse=mse, constantMse=baseline, psnr=float(-10*np.log10(mse)),
                        relativeMseReduction=1-mse/baseline, converged=bool(fit.success), iterations=fit.nit))
    draw.text((12,j*190+6),f'{label} / target vs. independent whole-neuron fit',fill='#242c34')
    panel.paste(target.resize((200,150), Image.Resampling.NEAREST).convert('RGB'),(12,j*190+28))
    result=Image.fromarray((pred.reshape(H,W)*255).clip(0,255).astype('uint8')).resize((200,150),Image.Resampling.NEAREST).convert('RGB')
    panel.paste(result,(228,j*190+28))
    draw.text((444,j*190+48),f'PSNR {records[-1]["psnr"]:.1f} dB\nMSE {mse:.4f}\nFlat MSE {baseline:.4f}',fill='#242c34')
    print(records[-1],flush=True)
panel.save(out/'whole-neuron-fit.png')
(out/'whole-neuron-fit.json').write_text(json.dumps(dict(resolution=[W,H],neurons=1500,pixelSupport=float(mask.mean()),
    operator='Length-sampled branch counts, density-normalized linear light; grayscale; independent bounded scalar per neuron',
    caveat='Unconstrained anatomical brightness experiment only. Not a stimulation controller, not the interactive depth renderer, not a proof of impossibility for other geometry.',results=records),indent=2))
pixel_index, neuron_index=np.nonzero(operator)
with (out/'image-operator.bin').open('wb') as handle:
    handle.write(np.array([W*H,1500,len(pixel_index)],dtype='<u4').tobytes())
    handle.write(pixel_index.astype('<u2').tobytes())
    handle.write(neuron_index.astype('<u2').tobytes())
    handle.write(operator[pixel_index,neuron_index].astype('<f4').tobytes())
(out/'controller-targets.json').write_text(json.dumps([dict(label='checker' if i==0 else f'doom-{frames[i-1]}',pixels=(np.asarray(t).astype(float).ravel()/255).round(5).tolist()) for i,t in enumerate(targets)]))
