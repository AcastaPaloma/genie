"""Matched scientific comparison of measured full-brain spike reconstructions.

All output columns come from simulated spikes. The source is a separate reference
column, never composited over an anatomical output or used as a runtime fallback.
"""
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
records = ROOT / ".cache/full-brain/validation"
labels = ["direct-image-v1", "subthreshold-image-loop-v1", "subthreshold-coverage-v1"]
titles = ["Source frame", "Main: tonic inputs", "Experimental: 700 seeds", "Experimental: 3,000 seeds"]
font_path = "/System/Library/Fonts/Supplemental/Arial.ttf"
font = ImageFont.truetype(font_path, 16)
small = ImageFont.truetype(font_path, 13)
result = Image.new("RGB", (1360, 1020), "#edf0f2")
draw = ImageDraw.Draw(result)
draw.text((16, 16), "FULL BRAIN: actual spike reconstructions", font=font, fill="#172a33")
draw.text((16, 42), "139,255 neurons; every released branch. These outputs still fail the recognition goal.", font=small, fill="#40525e")
draw.text((16, 62), "Experimental columns use engineered per-neuron shunting control and fixed excitatory seeds. All input arbors stay visible.", font=small, fill="#40525e")
for col, title in enumerate(titles):
    draw.text((16 + col * 336, 90), title, font=font, fill="#172a33")
for row, clip in enumerate(["map01", "map02", "map03"]):
    items = [json.loads((records / f"{label}-{clip}.json").read_text()) for label in labels]
    target = np.asarray(items[0]["target"], dtype=np.float32)
    for item in items:
        if not np.allclose(item["target"], target, atol=1e-6, rtol=0):
            raise ValueError("Comparison source frames differ")
    y = 118 + row * 292
    for col, values in enumerate([target] + [np.asarray(item["prediction"]) for item in items]):
        raster = np.clip(values.reshape(240, 320) * 255, 0, 255).astype(np.uint8)
        result.paste(Image.fromarray(raster).convert("RGB"), (16 + col * 336, y))
    draw.text((16, y + 248), f"{clip} · source 1.9 s", font=small, fill="#172a33")
    for col, item in enumerate(items, 1):
        draw.text((16 + col * 336, y + 248), f"Supported-region MSE {item['metrics']['maskedMse']:.5f}", font=small, fill="#40525e")
draw.text((16, 994), "Complete 320 × 240 anatomical operator. Empty projected regions cannot emit light. Lower error does not establish recognizable motion.", font=small, fill="#40525e")
output = ROOT / "public/data/full-brain-783/full-spike-controller-comparison.png"
result.save(output)
print(output)
