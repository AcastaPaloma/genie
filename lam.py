import torch
from torch import nn
from transformer import SpatioTemporalTransformerBlock
from torch.nn import functional as F
from cookbook import Cookbook

class LAMEncoder(nn.Module):
    def __init__(self, patch=8, img=64, channels=1, d_width=256, code_width=32, T=16, st_blocks=6, st_heads=8):
        super().__init__()
        self.patch, self.img, self.channels, self.d_width = patch, img, channels, d_width
        self.n_patches = (img // patch) ** 2       # 64
        patch_dim = patch * patch * channels       # 64
        self.patch_proj   = nn.Linear(patch_dim, d_width)
        self.spatial_pos  = nn.Parameter(torch.zeros(self.n_patches, d_width))
        self.temporal_pos = nn.Parameter(torch.zeros(T, d_width))
        self.blocks = nn.ModuleList([
            SpatioTemporalTransformerBlock(embedding_dim=d_width, num_heads=st_heads)
            for _ in range(st_blocks)
        ])
        self.to_code = nn.Linear(d_width, code_width)

    def forward(self, video):                    
        B, T, C, H, W = video.shape # [trajectories, frames, rgb, height, width]
        p = self.patch
        x = video.reshape(B, T, C, H // p, p, W // p, p) #[trajectories, frames, rgb, height/8, 8,  width/8, 8]
        x = x.permute(0, 1, 3, 5, 2, 4, 6) #[trajectories, frames, height/8, width/8, rgb, 8, 8]
        x = x.reshape(B, T, self.n_patches, -1) #[trajectories, frames, 64, 64]
        # STEP 2 — up to model width
        x = self.patch_proj(x)                             # [trajectories, frames ,64,256]
        # STEP 3 — positions
        x = x + self.spatial_pos[None, None, :, :]         # [1,1,64,256]
        x = x + self.temporal_pos[None, :T, None, :]       # [1,T,1,256]
        for block in self.blocks:
            x = block(x)                             # [B, T, 64, 256]  shape unchanged
        # STEP 5 — pool away the patch axis
        x = x.mean(dim=2)                            # [B, T, 256]
        # STEP 6 — down to code width
        x = self.to_code(x)                          # [B, T, 32]
        return x

class LAMDecoder(nn.Module):
    def __init__(self, patch=8, img=64, channels=1, d_width=256, code_width=32, T=16, st_heads=8, st_blocks=6):
        super().__init__()
        self.patch, self.img, self.channels, self.d_width = patch, img, channels, d_width
        patch_dim = patch * patch * channels
        self.n_patches = (img // patch) ** 2 
        self.patch_proj   = nn.Linear(patch_dim, d_width)
        self.spatial_pos  = nn.Parameter(torch.randn(self.n_patches, d_width) * 0.02)
        self.temporal_pos = nn.Parameter(torch.randn(T, d_width) * 0.02)
        self.blocks = nn.ModuleList([
                    SpatioTemporalTransformerBlock(embedding_dim=d_width, num_heads=st_heads)
                    for _ in range(st_blocks)
                ])
        self.action_proj = nn.Linear(code_width, d_width)
        self.to_pixels = nn.Linear(d_width, patch_dim)

    def forward(self, video, z_q):
        B, T, C, H, W = video.shape         # [16, 15, 1, 64, 64]
        p = self.patch # 8

        x = video.reshape(B, T, C, H // p, p, W // p, p) # [16, 15, 1, 8, 8, 8, 8]
        x = x.permute(0, 1, 3, 5, 2, 4, 6) # [16, 15, 8, 8, 1, 8, 8]
        x = x.reshape(B, T, self.n_patches, -1) # [16, 15, 64, 64]
        x = self.patch_proj(x) # [16, 15, 64, 256]
        x = x + self.spatial_pos[None, None, :, :] #  [64, 256]  → [1, 1, 64, 256]
        x = x + self.temporal_pos[None, :T, None, :]  #  [16, 256]  → [1, T, 1, 256]
        a = self.action_proj(z_q) # [16, 15, 32]  →  a  [16, 15, 256]
        a = a.unsqueeze(2) # [16, 15, 64, 256]
        x = x + a

        for block in self.blocks:
                    x = block(x)  # [16, 15, 64, 256]  shape unchanged
        
        x = self.to_pixels(x)  # [16, 15, 64, 256]  →  [16, 15, 64, 64]

        x = x.reshape(B, T, H // p, W // p, C, p, p)   # [16,15,8,8,1,8,8]
        x = x.permute(0, 1, 4, 2, 5, 3, 6)             # [16,15,1,8,8,8,8]
        x = x.reshape(B, T, C, H, W)                   # [16,15,1,64,64]

        return x

class LAM(nn.Module):
    def __init__(self, K=8, code_width=32, d_width=256, T=16, channels=1):
        super().__init__()
        self.K = K
        self.encoder  = LAMEncoder(code_width=code_width, d_width=d_width,
                                T=T, channels=channels)
        self.cookbook = Cookbook(K=K, code_width=code_width)
        self.decoder  = LAMDecoder(code_width=code_width, d_width=d_width,
                                T=T, channels=channels)

    def forward(self, video):
        z = self.encoder(video) # z    [16, 16, 32]     one 32-number vector per frame
        actions = z[:, 1:]
        z_q, idx, vq_loss = self.cookbook(actions) 
        past    = video[:, :-1]                        # frames 0..T-2
        target  = video[:, 1:]                         # frames 1..T-1

        pred = self.decoder(past, z_q)             # [B, T-1, C, 64, 64] predicted frames 1–14

        recon = F.mse_loss(pred, target)
        return pred, idx, recon + vq_loss, recon, vq_loss

    def train():
        pass


    