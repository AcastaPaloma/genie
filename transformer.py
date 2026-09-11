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

        # Mixing attention heads
        self.output_mix = nn.Linear(128, 128)

    def forward(self, x):
        """Patchify RGB videos shaped (batch, frames, 3, 64, 64).

        Returns spatially-positioned patch embeddings shaped
        (batch, frames, 16, 128).
        """
        def embed():
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

        frames_embedding = embed()

        qs1 = self.queries1(frames_embedding)
        ks1 = self.keys1(frames_embedding)
        vs1 = self.values1(frames_embedding)

        qs2 = self.queries2(frames_embedding)
        ks2 = self.keys2(frames_embedding)
        vs2 = self.values2(frames_embedding)

        qs3 = self.queries3(frames_embedding)
        ks3 = self.keys3(frames_embedding)
        vs3 = self.values3(frames_embedding)

        qs4 = self.queries4(frames_embedding)
        ks4 = self.keys4(frames_embedding)
        vs4 = self.values4(frames_embedding)

        qs5 = self.queries5(frames_embedding)
        ks5 = self.keys5(frames_embedding)
        vs5 = self.values5(frames_embedding)

        qs6 = self.queries6(frames_embedding)
        ks6 = self.keys6(frames_embedding)
        vs6 = self.values6(frames_embedding)

        qs7 = self.queries7(frames_embedding)
        ks7 = self.keys7(frames_embedding)
        vs7 = self.values7(frames_embedding)

        qs8 = self.queries8(frames_embedding)
        ks8 = self.keys8(frames_embedding)
        vs8 = self.values8(frames_embedding)

        am1 = torch.bmm(qs1, ks1.transpose(1, 2))
        am1 = am1 / (128 ** 0.5)
        am1 = torch.softmax(am1, dim=-1)
        tk1 = torch.bmm(am1, vs1)

        am2 = torch.bmm(qs2, ks2.transpose(1, 2))
        am2 = am2 / (128 ** 0.5)
        am2 = torch.softmax(am2, dim=-1)
        tk2 = torch.bmm(am2, vs2)

        am3 = torch.bmm(qs3, ks3.transpose(1, 2))
        am3 = am3 / (128 ** 0.5)
        am3 = torch.softmax(am3, dim=-1)
        tk3 = torch.bmm(am3, vs3)

        am4 = torch.bmm(qs4, ks4.transpose(1, 2))
        am4 = am4 / (128 ** 0.5)
        am4 = torch.softmax(am4, dim=-1)
        tk4 = torch.bmm(am4, vs4)

        am5 = torch.bmm(qs5, ks5.transpose(1, 2))
        am5 = am5 / (128 ** 0.5)
        am5 = torch.softmax(am5, dim=-1)
        tk5 = torch.bmm(am5, vs5)

        am6 = torch.bmm(qs6, ks6.transpose(1, 2))
        am6 = am6 / (128 ** 0.5)
        am6 = torch.softmax(am6, dim=-1)
        tk6 = torch.bmm(am6, vs6)

        am7 = torch.bmm(qs7, ks7.transpose(1, 2))
        am7 = am7 / (128 ** 0.5)
        am7 = torch.softmax(am7, dim=-1)
        tk7 = torch.bmm(am7, vs7)

        am8 = torch.bmm(qs8, ks8.transpose(1, 2))
        am8 = am8 / (128 ** 0.5)
        am8 = torch.softmax(am8, dim=-1)
        tk8 = torch.bmm(am8, vs8)

        # Ready for residual attention
        corr_tokens = self.output_mix(torch.cat((tk1, tk2, tk3, tk4, tk5, tk6, tk7, tk8), dim=-1))

        