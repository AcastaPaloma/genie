import torch
from torch import nn
import torch.nn.functional as F

class Cookbook(nn.Module):
    def __init__(self, K=1024, code_width=32, beta=0.25):
        super().__init__()
        self.K, self.code_width, self.beta = K, code_width, beta
        self.cookbook = nn.Embedding(K, code_width)
        self.cookbook.weight.data.uniform_(-1.0 / K, 1.0 / K)
        return

    def forward(self, z):
        z = z # [M, K, 32] -> [clips, num_choices, code_width]
        leading = z.shape[:-1] # save the starting state
        flat = z.reshape(-1, self.code_width)  # [M, 32]
        d = flat.unsqueeze(1) - self.cookbook.weight.unsqueeze(0)  # [M, K, 32]
        dist = d.pow(2).sum(dim=2)                                  # [M, K]
        idx = dist.argmin(dim=1)                                    # [M]
        z_q = self.cookbook(idx)                                    # [M, 32]

        codebook_loss   = F.mse_loss(z_q, flat.detach())            # rows → encoder outputs
        commitment_loss = F.mse_loss(flat, z_q.detach())            # encoder outputs → rows
        loss = codebook_loss + self.beta * commitment_loss
        z_q = flat + (z_q - flat).detach()                          # straight-through

        return z_q.reshape(*leading, self.code_width), idx.reshape(*leading), loss # z_q is the action in vector form (added to ), idx is the actual action output, loss is loss
    
    @torch.no_grad()
    def usage(self, idx):
        flat = idx.flatten()                                      # [4,16] → [64]
        counts = torch.bincount(flat, minlength=self.K).float()    # [K]
        p = counts / counts.sum().clamp(min=1)                     # [K], sums to 1
        perplexity = torch.exp(-(p * (p + 1e-10).log()).sum())     # scalar
        return counts, perplexity # counts is out of 8, sums to 1, perplexity is scalar, explaining the variety of choices

    # call this command like this:  counts, ppl = self.cookbook.usage(idx)