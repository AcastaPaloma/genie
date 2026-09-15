"""Rebuild the synaptic renderer's anatomical basis and sequence stimuli.

Uses only the shipped geometry and lossless dataset sprites. No anatomy is
warped, and the image basis contains one column per whole displayed neuron.
The 15-degree view was selected by a nine-pose geometry-only pilot.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'public/data'
CACHE = ROOT / '.cache/control-goal'
CACHE.mkdir(parents=True, exist_ok=True)

points = np.fromfile(DATA/'brain-points.bin', dtype='<f4').reshape(-1, 4)
normal = np.array([-np.sin(np.pi/12), 0., np.cos(np.pi/12)])
right = np.cross([0, 1, 0], normal)
right /= np.linalg.norm(right)
up = np.cross(normal, right)
projected = points[:, :3] @ np.stack([right, up], axis=1)
uv = (projected-[0, 25])/[300, 225]+.5
valid = ((uv >= 0) & (uv < 1)).all(axis=1)
xy = (uv[valid]*[40, 30]).astype(int)
pixel = (29-xy[:, 1])*40+xy[:, 0]
matrix = np.zeros((1200, 1500))
np.add.at(matrix, (pixel, points[valid, 3].astype(int)), 1)
density = matrix.sum(axis=1)
matrix /= np.maximum(density[:, None], 1)
rows, columns = np.nonzero(matrix)
operator = (np.array([1200, 1500, len(rows)], dtype='<u4').tobytes()
            + rows.astype('<u2').tobytes()+columns.astype('<u2').tobytes()
            + matrix[rows, columns].astype('<f4').tobytes())
(DATA/'synaptic-image-operator.bin').write_bytes(operator)
metadata = dict(resolution=[40, 30], normal=normal.tolist(), center=[0, 25],
                footprintMicrometers=[300, 225], cellMicrometers=7.5,
                pixelSupport=float((density > 0).mean()), neurons=1500,
                anatomySha256=hashlib.sha256((DATA/'brain-points.bin').read_bytes()).hexdigest(),
                operatorSha256=hashlib.sha256(operator).hexdigest(),
                method='Length-sampled real branches; one scalar per neuron; summed activity divided by projected branch density',
                cameraSelection='Geometry-only pilot: yaw and pitch in {-15, 0, 15} degrees; lowest mean log MSE on DOOM record 0 frames 0, 52, 100. Evaluation is in-sample engineering evidence, not held-out generalization.')
(DATA/'synaptic-image-operator.json').write_text(json.dumps(metadata, indent=2)+'\n')

manifest = json.loads((DATA/'clips.json').read_text())
starts = [40, 32, 0, 48]
clips = []
for clip, start in zip(manifest['clips'], starts, strict=True):
    sheet = Image.open(DATA/clip['file']).convert('RGB')
    frames = []
    for index in range(start, start+16):
        x, y = (index % 16)*160, (index // 16)*90
        image = sheet.crop((x+20, y, x+140, y+90)).resize((80, 60), Image.Resampling.NEAREST)
        rgb = np.asarray(image).astype(float)
        luma = rgb @ np.array([.2126, .7152, .0722])/255
        frames.append(dict(index=index, luminance=luma.ravel().tolist()))
    clips.append(dict(clip=clip['id'], frames=frames))
(CACHE/'validation-clips.json').write_text(json.dumps(clips, separators=(',', ':')))
print(f'Anatomical support: {metadata["pixelSupport"]:.2%}; prepared 64 sequence stimuli.')
