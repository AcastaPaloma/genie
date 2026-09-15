"""Matched targets: observed full-graph spikes versus an anatomical upper bound."""
import json
from pathlib import Path
import numpy as np
from scipy.sparse import load_npz
from PIL import Image, ImageDraw, ImageFont
from evaluate_full_anatomy import fit
from prepare_full_brain import OUT, SOURCE

basis = load_npz(SOURCE / "whole-arbor-320.npz")
panel = Image.new("RGB", (1024, 1000), "#edf0f2")
draw = ImageDraw.Draw(panel)
font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", 15)
small = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", 12)
draw.text((16, 14), "COMPLETE BRAIN — matched-frame control comparison", font=font, fill="#19262d")
draw.text((16, 40), "139,255 neurons; every released branch. Current synaptic output still fails the recognition goal.", font=small, fill="#44535d")
for x, title in zip([16, 352, 688], ["Source target", "Actual simulated spikes", "Independent brightness fit"]):
    draw.text((x, 68), title, font=font, fill="#19262d")
draw.text((688, 88), "Diagnostic only — NOT neural playback", font=small, fill="#a12c35")
records = []
for index, clip in enumerate(["map01", "map02", "map03"]):
    record = json.loads((SOURCE / "validation" / f"direct-image-v1-{clip}.json").read_text())
    target = np.array(record["target"]).reshape(240, 320)
    observed = np.array(record["prediction"]).reshape(240, 320)
    optimistic, bound = fit(basis, target, 1000)
    y = 112 + 284*index
    for x, values in zip([16, 352, 688], [target, observed, optimistic]):
        panel.paste(Image.fromarray(np.uint8(np.clip(values, 0, 1)*255)).convert("RGB"), (x, y))
    draw.text((16, y+248), f"{clip} · source {record['metrics']['time']:.1f} s", font=font, fill="#19262d")
    draw.text((352, y+248), f"Supported-region MSE {record['metrics']['maskedMse']:.5f}", font=small, fill="#19262d")
    draw.text((688, y+248), f"Supported-region MSE {bound['mse']:.5f}", font=small, fill="#19262d")
    records.append(dict(clip=clip, observed=record["metrics"], anatomicalDiagnostic=bound))
draw.text((16, 970), "Numerical comparison uses the complete 320 × 240 cable operator. Empty projected regions cannot display the source.", font=small, fill="#44535d")
panel.save(OUT / "full-controller-comparison.png")
(SOURCE / "validation" / "matched-control-comparison.json").write_text(json.dumps(records, indent=2)+"\n")
print(OUT / "full-controller-comparison.png", flush=True)
