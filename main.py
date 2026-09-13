"""Inspect the configured full models without allocating their parameter storage."""

from dataclasses import asdict
import json
import torch

from config import (TOKENIZER_ENCODER, TOKENIZER_DECODER, ACTION_MODEL, DYNAMICS,
                    TOKENIZER_TRAINING, DYNAMICS_TRAINING, IMAGE_SIZE,
                    SEQUENCE_LENGTH, FPS, MASKGIT_STEPS, SAMPLING_TEMPERATURE)
from vae import VAE
from lam import LAM
from dynamics_model import DynamicsModel


def main():
    with torch.device("meta"):
        models = {"tokenizer": VAE(), "lam": LAM(), "dynamics": DynamicsModel()}
    report = {
        "architecture": {name: asdict(config) for name, config in (
            ("tokenizer_encoder", TOKENIZER_ENCODER),
            ("tokenizer_decoder", TOKENIZER_DECODER),
            ("lam_encoder_and_decoder", ACTION_MODEL), ("dynamics", DYNAMICS))},
        "parameter_counts_this_implementation": {
            name: sum(p.numel() for p in model.parameters()) for name, model in models.items()
        },
        "video": {"height_width": IMAGE_SIZE, "frames": SEQUENCE_LENGTH, "fps": FPS},
        "tokenizer_training": asdict(TOKENIZER_TRAINING),
        "dynamics_training": asdict(DYNAMICS_TRAINING),
        "sampling": {"maskgit_steps": MASKGIT_STEPS, "temperature": SAMPLING_TEMPERATURE},
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
