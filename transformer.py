import torch
import torch.nn as nn


class Transformer(nn.Module):
    def __init__(self):
        super().__init__()

        # Video and patch configuration.
        self.frame_height = 64
        self.frame_width = 64
        self.channels = 3
        self.patch_height = 16
        self.patch_width = 16
        self.embedding_dim = 128

        # A 64 x 64 frame contains a 4 x 4 grid of 16 x 16 patches.
        self.patches_per_frame = 16
        self.patch_dim = 16 * 16 * 3  # 768 RGB values per patch

        # Turns each flattened image patch into one 128-D token.
        self.proj = nn.Linear(768, 128)

        # One learnable spatial position embedding for each of the 16 patches.
        # It broadcasts across both the batch and time dimensions.
        self.p_space = nn.Parameter(torch.zeros(1, 1, 16, 128))

        # Eight attention heads split the 128-D embedding into 8 x 16-D heads.
        self.queries1 = nn.Linear(128, 16)
        self.keys1 = nn.Linear(128, 16)
        self.values1 = nn.Linear(128, 16)

        self.queries2 = nn.Linear(128, 16)
        self.keys2 = nn.Linear(128, 16)
        self.values2 = nn.Linear(128, 16)

        self.queries3 = nn.Linear(128, 16)
        self.keys3 = nn.Linear(128, 16)
        self.values3 = nn.Linear(128, 16)

        self.queries4 = nn.Linear(128, 16)
        self.keys4 = nn.Linear(128, 16)
        self.values4 = nn.Linear(128, 16)

        self.queries5 = nn.Linear(128, 16)
        self.keys5 = nn.Linear(128, 16)
        self.values5 = nn.Linear(128, 16)

        self.queries6 = nn.Linear(128, 16)
        self.keys6 = nn.Linear(128, 16)
        self.values6 = nn.Linear(128, 16)

        self.queries7 = nn.Linear(128, 16)
        self.keys7 = nn.Linear(128, 16)
        self.values7 = nn.Linear(128, 16)

        self.queries8 = nn.Linear(128, 16)
        self.keys8 = nn.Linear(128, 16)
        self.values8 = nn.Linear(128, 16)

    def forward(self, x):
        """Patchify RGB videos shaped (batch, frames, 3, 64, 64).

        Returns spatially-positioned patch embeddings shaped
        (batch, frames, 16, 128).
        """
        batch_size, num_frames, channels, height, width = x.shape
        if (channels, height, width) != (3, 64, 64):
            raise ValueError(
                "Expected x with shape (batch, frames, 3, 64, 64), "
                f"but received {tuple(x.shape)}."
            )

        # (B, T, 3, 64, 64) -> (B, T, 16, 768)
        x = x.reshape(batch_size, num_frames, 3, 4, 16, 4, 16)
        x = x.permute(0, 1, 3, 5, 2, 4, 6)
        x = x.reshape(batch_size, num_frames, 16, 768)

        # (B, T, 16, 768) -> (B, T, 16, 128)
        X = self.proj(x) + self.p_space
        return X
