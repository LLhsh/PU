"""Conditional latent diffusion with a clean-state prediction objective (Sec. 3.3)."""

import math

import torch
from torch import nn
from torch.nn import functional as F


class TimeBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_channels=256):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.norm1 = nn.GroupNorm(8, out_channels)
        self.time = nn.Linear(time_channels, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(8, out_channels)

    def forward(self, x, time):
        x = F.silu(self.norm1(self.conv1(x)))
        x = x + self.time(F.silu(time))[:, :, None, None]
        return F.silu(self.norm2(self.conv2(x)))


class DenoisingUNet(nn.Module):
    """Channels [64, 128, 256, 512]; condition merges at the initial layer."""

    def __init__(self, latent_channels):
        super().__init__()
        self.register_buffer("frequencies", torch.exp(
            -math.log(10000) * torch.arange(64, dtype=torch.float32) / 64))
        self.time_mlp = nn.Sequential(nn.Linear(128, 256), nn.SiLU(), nn.Linear(256, 256))
        self.initial = TimeBlock(latent_channels + 64, 64)
        self.down = nn.ModuleList((TimeBlock(64, 128), TimeBlock(128, 256),
                                   TimeBlock(256, 512)))
        self.up = nn.ModuleList((TimeBlock(512 + 256, 256), TimeBlock(256 + 128, 128),
                                 TimeBlock(128 + 64, 64)))
        self.output = nn.Conv2d(64, latent_channels, 1)

    def forward(self, noisy, timesteps, condition):
        angles = timesteps.float()[:, None] * self.frequencies[None]
        time = self.time_mlp(torch.cat((angles.sin(), angles.cos()), dim=1))
        x = self.initial(torch.cat((noisy, condition), dim=1), time)
        skips = [x]
        for block in self.down:
            x = block(F.avg_pool2d(x, 2), time)
            skips.append(x)
        for block, skip in zip(self.up, reversed(skips[:-1])):
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            x = block(torch.cat((x, skip), dim=1), time)
        return self.output(x)


class LabelCodec(nn.Module):
    """Parameter-free spatial label codec; preserves one-hot class channels."""

    def __init__(self, latent_size):
        super().__init__()
        self.latent_size = latent_size

    def encode(self, labels):
        return F.adaptive_avg_pool2d(labels.detach().float(), self.latent_size)

    def decode(self, latent, output_size):
        return F.interpolate(latent, size=output_size, mode="bilinear", align_corners=False)


class UADR(nn.Module):
    def __init__(self, image_channels=1, num_classes=4, steps=50, latent_size=64):
        super().__init__()
        if steps < 2 or latent_size < 8 or latent_size % 8 or image_channels < 1 or num_classes < 2:
            raise ValueError("Invalid UADR dimensions or diffusion step count")
        self.steps = steps
        self.latent_size = latent_size
        self.image_channels = image_channels
        self.num_classes = num_classes
        self.codec = LabelCodec(latent_size)
        self.condition_encoder = nn.Sequential(
            nn.Conv2d(image_channels + 1, 32, 3, stride=2, padding=1),
            nn.GroupNorm(8, 32), nn.SiLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.GroupNorm(8, 64), nn.SiLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.GroupNorm(8, 64), nn.SiLU(),
        )
        self.denoiser = DenoisingUNet(num_classes)

        # Cosine cumulative variance schedule; index 0 represents paper t=1.
        grid = torch.linspace(0, steps, steps + 1, dtype=torch.float64)
        alpha_bar = torch.cos(((grid / steps + 0.008) / 1.008) * math.pi / 2).square()
        alpha_bar = alpha_bar / alpha_bar[0]
        betas = (1 - alpha_bar[1:] / alpha_bar[:-1]).clamp(max=0.999)
        alphas = 1 - betas
        cumulative = alphas.cumprod(dim=0)
        previous = torch.cat((torch.ones(1, dtype=torch.float64), cumulative[:-1]))
        self.register_buffer("alpha_bar", cumulative.float())
        self.register_buffer("posterior_variance", (betas * (1 - previous) /
                                                     (1 - cumulative)).float())
        self.register_buffer("posterior_z0", (betas * previous.sqrt() /
                                               (1 - cumulative)).float())
        self.register_buffer("posterior_zt", ((1 - previous) * alphas.sqrt() /
                                               (1 - cumulative)).float())

    def encode_labels(self, low_labels):
        # Sec. 4.2 specifies spatial compression, but no trainable codec objective.
        return self.codec.encode(low_labels)

    def check_inputs(self, image, uncertainty, low_labels):
        if image.ndim != 4 or image.shape[1] != self.image_channels:
            raise ValueError("UADR image must have shape [B,image_channels,H,W]")
        batch, _, height, width = image.shape
        if uncertainty.shape != (batch, 1, height, width):
            raise ValueError("UADR uncertainty must align with the original image")
        if low_labels.shape != (batch, self.num_classes, height, width):
            raise ValueError("UADR low-confidence one-hot labels must align with the original image")

    def encode_condition(self, image, uncertainty):
        # Xu, Uu and YL must have the same spatial coordinate system.
        condition = self.condition_encoder(torch.cat((image.detach(), uncertainty.detach()), dim=1))
        return F.interpolate(condition, (self.latent_size, self.latent_size),
                             mode="bilinear", align_corners=False)

    def q_sample(self, clean, timesteps, noise=None):
        if noise is None:
            noise = torch.randn_like(clean)
        alpha = self.alpha_bar[timesteps][:, None, None, None]
        return alpha.sqrt() * clean + (1 - alpha).sqrt() * noise

    def forward(self, image, uncertainty, low_labels):
        """Eq. (14): random-t clean-state prediction, updating only this branch."""
        self.check_inputs(image, uncertainty, low_labels)
        clean = self.encode_labels(low_labels)
        condition = self.encode_condition(image, uncertainty)
        timesteps = torch.randint(self.steps, (clean.shape[0],), device=clean.device)
        noisy = self.q_sample(clean, timesteps)
        prediction = self.denoiser(noisy, timesteps + 1, condition)
        return F.mse_loss(prediction, clean)

    @torch.no_grad()
    def rectify(self, image, uncertainty, low_labels, high_mask, weak_labels):
        """T -> 0 chain initialized from q(z_T | z_0), rather than pure noise."""
        self.check_inputs(image, uncertainty, low_labels)
        if high_mask.shape != weak_labels.shape or weak_labels.shape != (image.shape[0], *image.shape[-2:]):
            raise ValueError("Rectification masks and weak labels must have shape [B,H,W]")
        clean = self.encode_labels(low_labels)
        condition = self.encode_condition(image, uncertainty)
        batch = clean.shape[0]
        final_time = torch.full((batch,), self.steps - 1, dtype=torch.long, device=clean.device)
        state = self.q_sample(clean, final_time)
        for index in reversed(range(self.steps)):
            time = torch.full((batch,), index + 1, dtype=torch.long, device=state.device)
            predicted_clean = self.denoiser(state, time, condition)
            mean = self.posterior_z0[index] * predicted_clean + self.posterior_zt[index] * state
            state = mean
            if index:
                state = state + self.posterior_variance[index].sqrt() * torch.randn_like(state)
        decoded = self.codec.decode(state, image.shape[-2:])
        corrected = decoded.argmax(dim=1)
        # Complete the low-confidence correction with the unchanged high-confidence YH.
        return torch.where(high_mask, weak_labels, corrected)
