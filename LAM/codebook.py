import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import v2

'''
K rows (cookbook size)
code width = 32
decay
eps





'''

class Cookbook():
    def __init__(self, K=8, code_width=32, beta=0.25):
        super().__init__()
        self.K, self.code_width, self.beta = K, code_width, beta
        self.cookbook = nn.embedding(K, code_width)
        self.cookbook.weight.data.uniform_(-1.0 / K, 1.0 / K)
        return

    def forward(self, z, beta):
        leading = z.shape[:-1]
        flat = torch.flatten(z, self.code_width)

        d = flat.unsqueeze(1) - self.cookbook.weight.unsqueeze(0)   # [M, K, 32]
        dist = d.pow(2).sum(dim=2)

        idx = dist.argmin(dim=1)                 # was: argmin(flat) — wrong tensor
        z_q = self.cookbook(idx) 

        min = torch.argmin(flat, dim=1)
        drag = torch.mse(min)
        output = drag

        return output