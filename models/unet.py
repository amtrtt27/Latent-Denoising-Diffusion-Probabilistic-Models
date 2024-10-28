import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

from .unet_modules import TimeEmbedding, DownSample, UpSample, ResBlock


class UNet(nn.Module):
    def __init__(self, input_size, input_ch, T, ch, ch_mult, attn, num_res_blocks, dropout=0.0, conditional=False, c_dim=None):
        super().__init__()
        assert all([i < len(ch_mult) for i in attn]), 'attn index out of bound'
        
        self.input_size = input_size 
        self.input_ch = input_ch
        self.T = T 
        
        
        tdim = ch * 4
        self.time_embedding = TimeEmbedding(T, ch, tdim)

        self.stem = nn.Conv2d(input_ch, ch, kernel_size=3, stride=1, padding=1)
        self.downblocks = nn.ModuleList()
        chs = [ch]  # record output channel when dowmsample for upsample
        now_ch = ch
        for i, mult in enumerate(ch_mult):
            out_ch = ch * mult
            for _ in range(num_res_blocks):
                self.downblocks.append(ResBlock(
                    in_ch=now_ch, out_ch=out_ch, tdim=tdim,
                    dropout=dropout, attn=(i in attn), cross_attn=conditional and (i in attn), cdim=c_dim))
                now_ch = out_ch
                chs.append(now_ch)
            if i != len(ch_mult) - 1:
                self.downblocks.append(DownSample(now_ch))
                chs.append(now_ch)

        self.middleblocks = nn.ModuleList([
            ResBlock(now_ch, now_ch, tdim, dropout, attn=True, cross_attn=conditional, cdim=c_dim),
            ResBlock(now_ch, now_ch, tdim, dropout, attn=False),
        ])

        self.upblocks = nn.ModuleList()
        for i, mult in reversed(list(enumerate(ch_mult))):
            out_ch = ch * mult
            for _ in range(num_res_blocks + 1):
                self.upblocks.append(ResBlock(
                    in_ch=chs.pop() + now_ch, out_ch=out_ch, tdim=tdim,
                    dropout=dropout, attn=(i in attn), cross_attn=conditional and (i in attn), cdim=c_dim))
                now_ch = out_ch
            if i != 0:
                self.upblocks.append(UpSample(now_ch))
        assert len(chs) == 0

        self.head = nn.Sequential(
            nn.GroupNorm(32, now_ch),
            nn.SiLU(),
            nn.Conv2d(now_ch, input_ch, 1, stride=1, padding=0)
        )
        self.initialize()

    def initialize(self):
        init.xavier_uniform_(self.stem.weight)
        init.zeros_(self.stem.bias)
        init.xavier_uniform_(self.head[-1].weight, gain=1e-5)
        init.zeros_(self.head[-1].bias)

    def forward(self, x, t, c=None):
        # 1. time
        if not torch.is_tensor(t):
            t = torch.tensor([t], dtype=torch.long, device=x.device)
        elif torch.is_tensor(t) and len(t.shape) == 0:
            t = t[None].to(x.device)

        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        t = t * torch.ones(x.shape[0], dtype=t.dtype, device=t.device)

        # Timestep embedding
        temb = self.time_embedding(t)
        # Downsampling
        h = self.stem(x)
        hs = [h]
        for layer in self.downblocks:
            h = layer(h, temb, c)
            hs.append(h)
        # Middle
        for layer in self.middleblocks:
            h = layer(h, temb, c)
        # Upsampling
        for layer in self.upblocks:
            if isinstance(layer, ResBlock):
                h = torch.cat([h, hs.pop()], dim=1)
            h = layer(h, temb, c)
        h = self.head(h)

        assert len(hs) == 0
        return h