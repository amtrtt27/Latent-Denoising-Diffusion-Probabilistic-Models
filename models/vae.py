import torch 
import torch.nn as nn
from contextlib import contextmanager

from .vae_modules import Encoder, Decoder
from .vae_distributions import DiagonalGaussianDistribution


class VAE(nn.Module):
    # NOTE: do not change anything in __init__ function
    def __init__(self,
                 ### Encoder Decoder Related
                 double_z=True,
                 z_channels=3,
                 embed_dim=3,
                 resolution=256,
                 in_channels=3,
                 out_ch=3,
                 ch=128,
                 ch_mult=[1,2,4],  # num_down = len(ch_mult)-1
                 num_res_blocks=2):
        super(VAE, self).__init__()
        
        self.encoder = Encoder(in_channels=in_channels, ch=ch, out_ch=out_ch, num_res_blocks=num_res_blocks, z_channels=z_channels, ch_mult=ch_mult, resolution=resolution, double_z=double_z, attn_resolutions=[])
        self.decoder = Decoder(in_channels=in_channels, ch=ch, out_ch=out_ch, num_res_blocks=num_res_blocks, z_channels=z_channels, ch_mult=ch_mult, resolution=resolution, double_z=double_z, attn_resolutions=[])
        self.quant_conv = torch.nn.Conv2d(2*z_channels, 2*embed_dim, 1)
        self.post_quant_conv = torch.nn.Conv2d(embed_dim, z_channels, 1)

    @torch.no_grad()
    def encode(self, x):
        # TODO: Implemente the encode function transforms images into a sampled vector from
        h = None
        moments = None 
        
        # sample from Gaussian using re-parameterization trick
        posterior = DiagonalGaussianDistribution(moments)
        posterior = posterior.sample()
        return posterior

    @torch.no_grad()
    def decode(self, z):
        # TODO: reconstruct images from latent
        z = None 
        dec = None
        return dec

    def init_from_ckpt(self, path, ignore_keys=list()):
        sd = torch.load(path, map_location="cpu")["state_dict"]
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        keys = self.load_state_dict(sd, strict=False)
        print(f"Restored from {path}")
        print(keys)