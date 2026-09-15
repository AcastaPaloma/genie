"""Render measured LIF controller results, never its linear predictions."""
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

out = Path(__file__).resolve().parents[1] / "public/data"
targets = json.loads((out / "controller-targets.json").read_text())
calibration = json.loads((out / "stimulation-controller.json").read_text())
panel = Image.new("RGB", (760, 4 * 215 + 70), "#f8f9fa")
draw = ImageDraw.Draw(panel)
draw.text((16, 10), "Inverse stimulation experiment — actual full-network LIF responses", fill="#242c34")
draw.text((16, 28), "40 x 30 grayscale. Linear anatomical analysis renderer; not a browser screenshot.", fill="#616d79")

for row, (target, result) in enumerate(zip(targets, calibration["evaluation"], strict=True)):
    top = 62 + row * 215
    label = "Checkerboard" if row == 0 else f"DOOM frame {[1, 53, 101][row-1]}"
    for col, (title, values) in enumerate([
        ("Target", target["pixels"]),
        ("Constant stimulation", calibration["baseline"]),
        ("Controlled spiking result", result["actual"]),
    ]):
        x = 16 + col * 248
        draw.text((x, top), f"{label} / {title}" if col == 0 else title, fill="#242c34")
        pixels = (np.asarray(values).reshape(30, 40) * 255).clip(0, 255).astype("uint8")
        image = Image.fromarray(pixels).resize((224, 168), Image.Resampling.NEAREST)
        panel.paste(image.convert("RGB"), (x, top + 20))
    draw.text((512, top + 193), f'MSE {result["mse"]:.4f} / PSNR {result["psnr"]:.2f} dB', fill="#242c34")
panel.save(out / "stimulation-validation.png")
print(out / "stimulation-validation.png")
