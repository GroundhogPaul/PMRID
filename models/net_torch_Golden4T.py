#!/usr/bin/env python3
from pathlib import Path
import sys, os
curFolder = Path(__file__).resolve().parents[0]
if str(curFolder) not in sys.path:
    sys.path.append(str(curFolder))

import utilModels
from models.net_torch import NetworkBasic

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torchinfo import summary
import time
from thop import profile
from collections import OrderedDict

def conv(in_channels, out_channels, activation=nn.LeakyReLU(negative_slope=0.2)):
    """Applies a 3x3 conv layer with optional activation."""
    layers = [
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
    ]
    if activation is not None:
        layers.append(activation)
    return nn.Sequential(*layers)

class ConvBlock(nn.Module):
    """Applies 3x conv layers with the same number of channels."""

    def __init__(self, ChIn, ChOut):
        super(ConvBlock, self).__init__()
        self.conv1 = conv(ChIn, ChOut)
        self.conv2 = conv(ChOut, ChOut)
        self.conv3 = conv(ChOut, ChOut)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return x


def downsample_2x(x):
    """Applies a 2x spatial downsample via max pooling with 'same' padding behavior."""
    # To emulate TensorFlow's 'same' padding, we manually pad if the input size is odd.
    # This ensures output size = ceil(input_size / 2).
    batch, _, h, w = x.shape
    pad_h = (2 - h % 2) % 2  # 1 if h is odd, else 0
    pad_w = (2 - w % 2) % 2
    if pad_h or pad_w:
        x = F.pad(x, (0, pad_w, 0, pad_h), mode='constant', value=0)
    return F.max_pool2d(x, kernel_size=2, stride=2)


def upsample_2x(x):
    """Applies a 2x spatial upsample via bilinear interpolation."""
    return F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)


class Golden4T(nn.Module):
    def __init__(self):
        super(Golden4T, self).__init__()

        # ----- Encoder ----- #
        self.enc_channels = [8, 32, 64, 128, 256]
        lstEncoder = []
        for i, ChIn in enumerate(self.enc_channels[:-1]):
            ChOut = self.enc_channels[i+1]
            lstEncoder.append(ConvBlock(ChIn, ChOut))
        self.enc_blocks = nn.ModuleList(lstEncoder)

        # ----- Decoder ----- #
        self.dec_channels_in = [512+256, 256+128, 128+64, 64+32]
        self.dec_channels_out = [256, 128, 64, 32]
        lstDecoder = []
        for i, ChIn in enumerate(self.dec_channels_in):
            ChOut = self.dec_channels_out[i]
            lstDecoder.append(ConvBlock(ChIn, ChOut))
        self.dec_blocks = nn.ModuleList(lstDecoder)

        # Final convolution (no activation) that outputs residual
        self.residual_conv = nn.Conv2d(self.dec_channels_out[-1], 4, kernel_size=3, padding=1)

    def forward(self, noisy_img, variance):
        """
        Args:
            noisy_img: Tensor of shape [B, 4, H, W]
            variance:   Tensor of shape [B, 4, H, W]

        Returns:
            Denoised image of shape [B, 4, H, W]
        """
        # Ensure input shapes are compatible
        assert noisy_img.shape == variance.shape, "noisy_img and variance must have same shape"

        # Concatenate along channel dimension -> [B, 8, H, W]
        features = torch.cat([noisy_img, variance], dim=1)

        skip_connections = []

        # Encoder
        for i, num_channels in enumerate(self.enc_channels[1:]):
            # print("encoder: ", i)
            features = self.enc_blocks[i](features)
            skip_connections.append(features)
            features = downsample_2x(features)

        bottleneck_conv = ConvBlock(256, 512).to(features.device)
        features = bottleneck_conv(features)

        # Decoder
        for i, num_channels in enumerate(self.dec_channels_in):
            # print("decoder: ", i)
            features = upsample_2x(features)
            # Concatenate with the corresponding skip connection (pop from end)
            skip = skip_connections.pop()
            # Ensure spatial dimensions match (due to possible odd-size padding)
            if features.shape[2:] != skip.shape[2:]:
                # Align sizes: skip may be larger; interpolate features or crop skip?
                # Usually they match due to same-padding pooling, but if not, interpolate features.
                features = F.interpolate(features, size=skip.shape[2:], mode='bilinear', align_corners=False)
            features = torch.cat([features, skip], dim=1)
            features = self.dec_blocks[i](features)

        residual = self.residual_conv(features)

        denoised = noisy_img + residual
        return denoised

if __name__ == "__main__":
    net, img = Golden4T(), torch.randn(1, 4, 64, 64, device=torch.device('cpu'), dtype=torch.float32)
    out = net(img, img)

    # summary(net, input_size=(1, 8, 64, 64))
    flops, params = profile(net, inputs=[img, img])
    gflops = flops/1e9
    imgSizeM = img.shape[1]*img.shape[2]*img.shape[3]/1e6
    print(f"FLOPs: {gflops:.2f}G, Params: {params/1e6:.2f}M, gFlops per 1M pixel = {gflops/imgSizeM:.2f}G")