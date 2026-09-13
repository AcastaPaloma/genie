"""Full Genie 1 defaults, not the smaller CoinRun/scaling experiments.

Source: https://arxiv.org/html/2402.15391v1 (Tables 5, 7–9, 12).
Unspecified implementation choices are called out in HYPERPARAMETERS.md.
"""

from dataclasses import dataclass
import math

from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR


@dataclass(frozen=True)
class TransformerConfig:
    width: int
    layers: int
    heads: int
    head_dim: int
    qk_norm: bool = False
    # The paper does not report the FFN expansion or dropout rates.
    ffn_expansion: float = 4.0
    dropout: float = 0.0


TOKENIZER_ENCODER = TransformerConfig(512, 12, 8, 64)
TOKENIZER_DECODER = TransformerConfig(1024, 20, 16, 64)
# Table 5 omits head_dim; 1024 / 16 = 64 is our choice.
ACTION_MODEL = TransformerConfig(1024, 20, 16, 64)
DYNAMICS = TransformerConfig(5120, 48, 36, 128, qk_norm=True)

VIDEO_CODES = 1024
ACTION_CODES = 8
CODE_WIDTH = 32
VIDEO_PATCH = 4
ACTION_PATCH = 16
SEQUENCE_LENGTH = 16
FPS = 10
CHANNELS = 3
# Paper dataset: 160x90. Bottom padding to 96 is our implementation choice.
CONTENT_SIZE = (90, 160)  # (height, width)
IMAGE_SIZE = (96, 160)
VQ_BETA = 0.25  # Standard VQ-VAE choice, not specified in the Genie tables.


@dataclass(frozen=True)
class TrainingConfig:
    max_lr: float
    min_lr: float
    warmup_steps: int
    steps: int
    global_batch_size: int
    betas: tuple[float, float] = (0.9, 0.9)
    weight_decay: float = 1e-4
    precision: str = "bfloat16"


# Table 6 reports both 64 and 384; choose its larger tokenizer batch.
# Table 8 really lists equal max/min LR; do not silently "correct" it.
TOKENIZER_TRAINING = TrainingConfig(3e-4, 3e-4, 10_000, 300_000, 384)
DYNAMICS_TRAINING = TrainingConfig(3e-5, 3e-6, 5_000, 125_000, 512)
# LAM co-training uses the same optimizer schedule here (implementation choice).
LAM_TRAINING = DYNAMICS_TRAINING
MASK_RATE_RANGE = (0.5, 1.0)
MASKGIT_STEPS = 25
SAMPLING_TEMPERATURE = 2.0


def build_optimizer(model: nn.Module, config: TrainingConfig):
    """Return AdamW and a warmup/cosine scheduler; step scheduler after optimizer.

    Batch size is GLOBAL, across devices and accumulated microbatches. Precision
    must be applied with autocast by the training loop, not by casting token IDs.
    """
    optimizer = AdamW(
        (p for p in model.parameters() if p.requires_grad),
        lr=config.max_lr, betas=config.betas, weight_decay=config.weight_decay,
    )

    def multiplier(step):
        if step < config.warmup_steps:
            return (step + 1) / config.warmup_steps
        progress = min(1.0, (step - config.warmup_steps) /
                       max(1, config.steps - config.warmup_steps))
        ratio = config.min_lr / config.max_lr
        return ratio + (1 - ratio) * (1 + math.cos(math.pi * progress)) / 2

    return optimizer, LambdaLR(optimizer, multiplier)
