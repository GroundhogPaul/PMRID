# the net for SamSung Remosaic (guessed from .so)
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


class Network_HM6Nona(nn.Module):
    def __init__(self):
        super().__init__()

        # ----- Encoder L0 ----- #
        self.Encoder0conv0 = Conv2D(9, 9, kernel_size=3, stride=1, padding=1, is_seperable=False)
        self.Encoder0PReLu0 = nn.PReLU(num_parameters = 9)
        self.Encoder0conv1 = Conv2D(9, 9, kernel_size=3, stride=1, padding=1, is_seperable=False)
        self.Encoder0PReLu1 = nn.PReLU(num_parameters = 9)

        # ----- Encoder L1 ----- #
        self.Encoder1conv0 = Conv2D(9, 18, kernel_size=3, stride=2, padding=1, is_seperable=False)
        self.Encoder1conv1 = Conv2D(18, 18, kernel_size=3, stride=1, padding=1, is_seperable=False)
        self.Encoder1PReLu1 = nn.PReLU(num_parameters = 18)

        # ----- Encoder L2 ----- #
        self.Encoder2conv0 = Conv2D(18, 27, kernel_size=3, stride=2, padding=1, is_seperable=False)
        self.Encoder2conv1 = Conv2D(27, 27, kernel_size=3, stride=1, padding=1, is_seperable=False)
        self.Encoder2PReLu1 = nn.PReLU(num_parameters = 27)

        # ----- Decoder L1 ----- #
        self.Decoder1Tconv = nn.ConvTranspose2d(27, 18, kernel_size=3, stride=2, padding=1, output_padding=1)   # 避免输出尺寸奇偶歧义，通常设为1
        self.Decoder1PReLu0 = nn.PReLU(num_parameters = 18)
        self.Decoder1conv = Conv2D(18, 18, kernel_size=3, stride=1, padding=1, is_seperable=False)
        self.Decoder1PReLu1 = nn.PReLU(num_parameters = 18)

        # ----- Decoder L0 ----- #
        self.Decoder0Tconv = nn.ConvTranspose2d(18, 9, kernel_size=3, stride=3, padding=1, output_padding=1)   # 避免输出尺寸奇偶歧义，通常设为1
        self.Decoder0PReLu0 = nn.PReLU(num_parameters = 9)
        self.Decoder0conv1 = Conv2D(9, 9, kernel_size=3, stride=1, padding=1, is_seperable=False)
        self.Decoder0PReLu1 = nn.PReLU(num_parameters = 9)
        self.Decoder0conv2 = Conv2D(9, 4, kernel_size=3, stride=1, padding=1, is_seperable=False)
        
    def forward(self, x, var=None):

        # ----- Encoder L0 ----- #
        Encoder0conv0 = self.Encoder0conv0(x)
        Encoder0conv0PReLu = self.Encoder0PReLu0(Encoder0conv0 )
        Encoder0conv1 = self.Encoder0conv1(Encoder0conv0)
        Encoder0conv1PRelu = self.Encoder0PReLu1(Encoder0conv1)
        Encoder0 = Encoder0conv0PReLu + Encoder0conv1PRelu 

        # ----- Encoder L1 ----- #
        Encoder1conv0 = self.Encoder1conv0(Encoder0)
        Encoder1conv1 = self.Encoder1conv1(Encoder1conv0)
        Encoder1conv1PReLu = self.Encoder1PReLu1(Encoder1conv1)
        Encoder1 = Encoder1conv0 + Encoder1conv1PReLu

        # ----- Encoder L2 ----- #
        Encoder2conv0 = self.Encoder2conv0(Encoder1)
        Encoder2conv1 = self.Encoder2conv1(Encoder2conv0)
        Encoder2conv1PReLu = self.Encoder2PReLu1(Encoder2conv1)
        Encoder1 = Encoder2conv0 + Encoder2conv1PReLu 

        # ----- Decoder l1 ----- #
        Decoder1Tconv = self.Decoder1Tconv(Encoder1)
        Decoder1TconvPRelu = self.Decoder1PReLu0(Decoder1Tconv)
        Decoder1TconvRes = Decoder1TconvPRelu + Encoder1conv0
        Decoder1conv = self.Decoder1conv(Decoder1TconvRes)
        Decoder1convPReLu = self.Decoder1PReLu1(Decoder1conv)
        Decoder1 = Decoder1convPReLu + Decoder1TconvRes

        # ----- Decoder l0 ----- #
        Decoder0Tconv = self.Decoder0Tconv(Decoder1)
        Decoder0TconvPRelu = self.Decoder0PReLu0(Decoder0Tconv)
        Decoder0TconvRes = Decoder0TconvPRelu # + Encoder0conv0
        Decoder0conv = self.Decoder0conv1(Decoder0TconvRes)
        Decoder0convPReLu = self.Decoder0PReLu1(Decoder0conv)
        Decoder0 = Decoder0convPReLu + Decoder0TconvRes

        # ----- Output l0 ----- #
        Out = self.Decoder0conv2(Decoder0)

        return Out 

if __name__ == "__main__":
    net, img = Network_HM6Nona(), torch.randn(1, 9, 576, 576, device=torch.device('cpu'), dtype=torch.float32)
    out = net(img)

    # summary(net, input_size=(1, 8, 64, 64))
    flops, params = profile(net, inputs=[img])
    gflops = flops/1e9
    imgSizeM = img.shape[1]*img.shape[2]*img.shape[3]/1e6
    print(f"FLOPs: {gflops:.2f}G, Params: {params/1e6:.2f}M, gFlops per 1M pixel = {gflops/imgSizeM:.2f}G")