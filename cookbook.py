"""Vector quantization shared by the video tokenizer and latent action model."""

import torch
from torch import nn
import torch.nn.functional as F

from config import VIDEO_CODES, CODE_WIDTH, VQ_BETA


class Cookbook(nn.Module):
    def __init__(self, K=VIDEO_CODES, code_width=CODE_WIDTH, beta=VQ_BETA,
                 *, search_chunk_size=4096):
        super().__init__()
        if min(K, code_width, search_chunk_size) < 1 or beta < 0:
            raise ValueError("Invalid codebook size, width, beta or chunk size")
        self.K, self.code_width, self.beta = K, code_width, beta
        self.search_chunk_size = search_chunk_size
        self.cookbook = nn.Embedding(K, code_width)
        nn.init.uniform_(self.cookbook.weight, -1.0 / K, 1.0 / K)

    def forward(self, z):
        if z.shape[-1] != self.code_width or not z.is_floating_point() or z.numel() == 0:
            raise ValueError("Expected nonempty floating-point code vectors of configured width")
        leading = z.shape[:-1]
        flat = z.reshape(-1, self.code_width)
        # Chunked squared distances avoid the former (all_tokens,K,32) tensor.
        # Keep nearest-neighbor search in float32, including under bf16 autocast.
        with torch.no_grad(), torch.autocast(device_type=z.device.type, enabled=False):
            codes = self.cookbook.weight.float()
            code_norms = codes.square().sum(-1)
            indices = []
            for chunk in flat.detach().split(self.search_chunk_size):
                chunk = chunk.float()
                distances = chunk.square().sum(-1, keepdim=True) + code_norms - 2 * chunk @ codes.T
                indices.append(distances.argmin(-1))
            idx = torch.cat(indices)
        quantized = self.cookbook(idx)
        codebook_loss = F.mse_loss(quantized.float(), flat.detach().float())
        commitment_loss = F.mse_loss(flat.float(), quantized.detach().float())
        loss = codebook_loss + self.beta * commitment_loss
        z_q = flat + (quantized.to(flat.dtype) - flat).detach()
        return z_q.reshape(*leading, self.code_width), idx.reshape(*leading), loss

    @torch.no_grad()
    def usage(self, idx):
        counts = torch.bincount(idx.flatten(), minlength=self.K).float()
        p = counts / counts.sum().clamp(min=1)
        perplexity = torch.exp(-(p * (p + 1e-10).log()).sum())
        return counts, perplexity
