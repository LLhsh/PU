"""Main segmentation U-Net. Auxiliary PU-Diff modules are training-only."""

import torch
from torch import nn
from torch.nn import functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=0.0):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels), nn.LeakyReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels), nn.LeakyReLU(inplace=True),
        )

    def forward(self, x):
        return self.layers(x)


class UpBlock(nn.Module):
    def __init__(self, in_channels, skip_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, skip_channels, 2, stride=2)
        self.conv = ConvBlock(2 * skip_channels, skip_channels)

    def forward(self, x, skip):
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.conv(torch.cat((skip, x), dim=1))


class UNet(nn.Module):
    feature_channels = 16

    def __init__(self, in_chns=1, class_num=4):
        super().__init__()
        channels = (16, 32, 64, 128, 256)
        dropouts = (0.05, 0.1, 0.2, 0.3, 0.5)
        self.encoder = nn.ModuleList()
        for index, width in enumerate(channels):
            self.encoder.append(ConvBlock(in_chns if index == 0 else channels[index - 1],
                                          width, dropouts[index]))
        self.decoder = nn.ModuleList(
            UpBlock(channels[index], channels[index - 1]) for index in range(4, 0, -1)
        )
        self.classifier = nn.Conv2d(channels[0], class_num, 3, padding=1)

    def forward(self, x, return_features=False):
        skips = []
        for index, block in enumerate(self.encoder):
            if index:
                x = F.max_pool2d(x, 2)
            x = block(x)
            skips.append(x)
        for block, skip in zip(self.decoder, reversed(skips[:-1])):
            x = block(x, skip)
        # F is the last feature map before the segmentation classifier (Sec. 3.2).
        logits = self.classifier(x)
        return (logits, x) if return_features else logits
