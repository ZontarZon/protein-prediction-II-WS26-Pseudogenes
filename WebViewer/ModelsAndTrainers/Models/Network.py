import torch
import torch.nn as nn
import math



# ============================================================
# Diffusion
# ============================================================

class Diffusion:

    def __init__( self, image_size, timesteps=1000, device="cpu" ):
        self.image_size = image_size
        self.timesteps = timesteps

        # Linear noise schedule
        self.beta = torch.linspace( 1e-4, 0.02, timesteps, device=device )

        self.alpha = 1.0 - self.beta

        self.alpha_bar = torch.cumprod( self.alpha, dim=0 )

    def to(self, device):
        self.beta = self.beta.to(device)
        self.alpha = self.alpha.to(device)
        self.alpha_bar = self.alpha_bar.to(device)
        return self

    def add_noise(self, images, timestep):

        noise = torch.randn_like(images)

        alpha_bar = self.alpha_bar[timestep]

        alpha_bar = alpha_bar.view( -1, 1, 1, 1 )

        noisy_images = ( torch.sqrt(alpha_bar) * images + torch.sqrt(1 - alpha_bar) * noise )

        return noisy_images, noise
