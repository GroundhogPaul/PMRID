#!/usr/bin/env python3
from pathlib import Path
import sys, os
curFolder = Path(__file__).resolve().parents[0]
if str(curFolder) not in sys.path:
    sys.path.append(str(curFolder))

import utilModels
from models.net_torch import NetworkBasic, Conv2D, EncoderStage_3x3, DecoderBlock, DecoderStage

import numpy as np
import torch
import torch.nn as nn
import time
from torchinfo import summary
from thop import profile
from collections import OrderedDict

class Network_2400M_BiLinear(NetworkBasic):
    def __init__(self, mode):
        super().__init__(mode)

        if self.mode == 'KSigma':
            self.conv0 = Conv2D(in_channels=4, out_channels=16, kernel_size=3, padding=1, stride=1, is_seperable=False, has_relu=True)
        elif self.mode == 'Concat':
            self.conv0 = Conv2D(in_channels=8, out_channels=16, kernel_size=3, padding=1, stride=1, is_seperable=False, has_relu=True)
        else:
            assert False

        self.enc1 = EncoderStage_3x3(in_channels=16, out_channels=32, num_blocks=2) # kernel size = 3
        self.enc2 = EncoderStage_3x3(in_channels=32, out_channels=32, num_blocks=2) # kernel size = 3
        self.enc3 = EncoderStage_3x3(in_channels=32, out_channels=32, num_blocks=2) # kernel size = 3

        self.encdec = Conv2D(in_channels=32, out_channels=32, kernel_size=3, padding=1, stride=1, is_seperable=True, has_relu=True)
        
        self.dec1 = DecoderStage(in_channels=32, skip_in_channels=32, out_channels=32, upsample='Bilinear')
        self.dec2 = DecoderStage(in_channels=32, skip_in_channels=32, out_channels=16, upsample='Bilinear')
        self.dec3 = DecoderStage(in_channels=16, skip_in_channels=16, out_channels=16, upsample='Bilinear')

        self.out0 = DecoderBlock(in_channels=16, out_channels=16, kernel_size=3)
        self.out1 = Conv2D(in_channels=16, out_channels=4, kernel_size=3, stride=1, padding=1, is_seperable=False, has_relu=False)
        
    def forward(self, inp, var=None):
        if self.mode == 'KSigma':
            x = inp
        elif self.mode == 'Concat':
            x = torch.cat([inp, var], dim=1)
        else:
            assert False
        
        conv0 = self.conv0(x)
        conv1 = self.enc1(conv0)
        conv2 = self.enc2(conv1)
        conv3 = self.enc3(conv2)

        conv4 = self.encdec(conv3)

        up2 = self.dec1((conv4, conv2))
        up1 = self.dec2((up2, conv1))
        x = self.dec3((up1, conv0))

        x = self.out0(x)
        x = self.out1(x)

        pred = inp + x
        return pred

if __name__ == "__main__":
    net, img = Network_2400M_BiLinear(mode='Concat'), torch.randn(1, 4, 64, 64, device=torch.device('cpu'), dtype=torch.float32)
    # out = net(img)

    # summary(net, input_size=(1, 8, 64, 64))
    flops, params = profile(net, inputs=([img, img]))
    gflops = flops/1e9
    imgSizeM = img.shape[1]*img.shape[2]*img.shape[3]/1e6
    print(f"FLOPs: {gflops:.2f}G, Params: {params/1e6:.2f}M, gFlops per 1M pixel = {gflops/imgSizeM:.2f}G")