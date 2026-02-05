#!/usr/bin/env python3
import torch
import torch.nn as nn
from thop import profile
from torchinfo import summary
from collections import OrderedDict

import numpy as np

class NetworkBasic(nn.Module):

    def __init__(self):
        super().__init__()
    
    def load_CKPT(self, sCKPT: str, device: torch.device):
        checkpoint = torch.load(sCKPT, map_location=device)
        print("Checkpoint类型:", type(checkpoint))
        print("Checkpoint键:", checkpoint.keys() if isinstance(checkpoint, dict) else "不是字典") 

        # 获取状态字典
        if isinstance(checkpoint, dict):
            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint
        else:
            state_dict = checkpoint.state_dict()

        # 打印模型和状态字典的键进行比较
        # print("模型键:", self.state_dict().keys())
        # print("状态字典键:", state_dict.keys())

        # assert set(self.state_dict().keys()) == set(state_dict.keys()), "模型键与状态字典键不匹配"
        self.load_state_dict(state_dict)
        print(f"Loaded model weights from {sCKPT}.")

        return

    def load_PTH(self, sPTH: str, device: torch.device):
        checkpoint = torch.load(sPTH, weights_only=False)
        print("Checkpoint类型:", type(checkpoint))
        print("Checkpoint键:", checkpoint.keys() if isinstance(checkpoint, dict) else "不是字典") 

        self.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded model weights from {sPTH}.")

        return
    

def Conv2D(
        in_channels: int, out_channels: int,
        kernel_size: int, stride: int, padding: int,
        is_seperable: bool = False, has_relu: bool = False,
):
    modules = OrderedDict()

    if is_seperable:
        modules['depthwise'] = nn.Conv2d(
            in_channels, in_channels, kernel_size, stride, padding,
            groups=in_channels, bias=False,
        )
        modules['pointwise'] = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=1, stride=1, padding=0, bias=True,
        )
    else:
        modules['conv'] = nn.Conv2d(
            in_channels, out_channels, kernel_size, stride, padding,
            bias=True,
        )
    if has_relu:
        modules['relu'] = nn.ReLU()

    return nn.Sequential(modules)

class EncoderBlock(nn.Module):

    def __init__(self, in_channels: int, mid_channels: int, out_channels: int, stride: int = 1):
        super().__init__()

        self.conv1 = Conv2D(in_channels, mid_channels, kernel_size=5, stride=stride, padding=2, is_seperable=True, has_relu=True)
        self.conv2 = Conv2D(mid_channels, out_channels, kernel_size=5, stride=1, padding=2, is_seperable=True, has_relu=False)

        self.proj = (
            nn.Identity()
            if stride == 1 and in_channels == out_channels else
            Conv2D(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, is_seperable=True, has_relu=False)
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        proj = self.proj(x)

        x = self.conv1(x)
        x = self.conv2(x)

        x = x + proj
        return self.relu(x)


def EncoderStage(in_channels: int, out_channels: int, num_blocks: int):

    blocks = [
        EncoderBlock(
            in_channels=in_channels,
            mid_channels=out_channels//4,
            out_channels=out_channels,
            stride=2,
        )
    ]
    for _ in range(num_blocks-1):
        blocks.append(
            EncoderBlock(
                in_channels=out_channels,
                mid_channels=out_channels//4,
                out_channels=out_channels,
                stride=1,
            )
        )

    return nn.Sequential(*blocks)


class DecoderBlock(nn.Module):

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3):
        super().__init__()

        padding = kernel_size // 2
        self.conv0 = Conv2D(
            in_channels, out_channels, kernel_size=kernel_size, padding=padding,
            stride=1, is_seperable=True, has_relu=True,
        )
        self.conv1 = Conv2D(
            out_channels, out_channels, kernel_size=kernel_size, padding=padding,
            stride=1, is_seperable=True, has_relu=False,
        )

    def forward(self, x):
        inp = x
        x = self.conv0(x)
        x = self.conv1(x)
        x = x + inp
        return x


class DecoderStage(nn.Module):

    def __init__(self, in_channels: int, skip_in_channels: int, out_channels: int):
        super().__init__()

        self.decode_conv = DecoderBlock(in_channels, in_channels, kernel_size=3)
        self.upsample = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2, padding=0)
        self.proj_conv = Conv2D(skip_in_channels, out_channels, kernel_size=3, stride=1, padding=1, is_seperable=True, has_relu=True)
        # M.init.msra_normal_(self.upsample.weight, mode='fan_in', nonlinearity='linear')

    def forward(self, inputs):
        inp, skip = inputs

        x = self.decode_conv(inp)
        x = self.upsample(x)
        y = self.proj_conv(skip)
        return x + y

class NetworkPMRID(NetworkBasic):

    def __init__(self, ChRatio):
        super().__init__()

        assert ChRatio == 0.5 or ChRatio == 1

        self.conv0 = Conv2D(in_channels=4, out_channels=16, kernel_size=3, padding=1, stride=1, is_seperable=False, has_relu=True)
        self.enc1 = EncoderStage(in_channels=16, out_channels=int(64*ChRatio), num_blocks=2)
        self.enc2 = EncoderStage(in_channels=int(64*ChRatio), out_channels=int(128*ChRatio), num_blocks=2)
        self.enc3 = EncoderStage(in_channels=int(128*ChRatio), out_channels=int(256*ChRatio), num_blocks=4)
        self.enc4 = EncoderStage(in_channels=int(256*ChRatio), out_channels=int(512*ChRatio), num_blocks=4)

        self.encdec = Conv2D(in_channels=int(512*ChRatio), out_channels=int(64*ChRatio), kernel_size=3, padding=1, stride=1, is_seperable=True, has_relu=True)
        self.dec1 = DecoderStage(in_channels=int(64*ChRatio), skip_in_channels=int(256*ChRatio), out_channels=int(64*ChRatio))
        self.dec2 = DecoderStage(in_channels=int(64*ChRatio), skip_in_channels=int(128*ChRatio), out_channels=int(32*ChRatio))
        self.dec3 = DecoderStage(in_channels=int(32*ChRatio), skip_in_channels=int(64*ChRatio), out_channels=int(32*ChRatio))
        self.dec4 = DecoderStage(in_channels=int(32*ChRatio), skip_in_channels=int(16), out_channels=16)

        self.out0 = DecoderBlock(in_channels=16, out_channels=16, kernel_size=3)
        self.out1 = Conv2D(in_channels=16, out_channels=4, kernel_size=3, stride=1, padding=1, is_seperable=False, has_relu=False)

    def forward(self, inp):

        conv0 = self.conv0(inp)
        conv1 = self.enc1(conv0)
        conv2 = self.enc2(conv1)
        conv3 = self.enc3(conv2)
        conv4 = self.enc4(conv3)

        conv5 = self.encdec(conv4)

        up3 = self.dec1((conv5, conv3))
        up2 = self.dec2((up3, conv2))
        up1 = self.dec3((up2, conv1))
        x = self.dec4((up1, conv0))

        x = self.out0(x)
        x = self.out1(x)

        pred = inp + x
        return pred

class NetworkPMRID2G(NetworkPMRID):

    def __init__(self):
        super().__init__()
        ChRatio = 0.5

        self.conv0 = Conv2D(in_channels=4, out_channels=16, kernel_size=3, padding=1, stride=1, is_seperable=False, has_relu=True)
        self.enc1 = EncoderStage(in_channels=16, out_channels=int(64*ChRatio), num_blocks=2)
        self.enc2 = EncoderStage(in_channels=int(64*ChRatio), out_channels=int(64*ChRatio), num_blocks=2)
        self.enc3 = EncoderStage(in_channels=int(64*ChRatio), out_channels=int(64*ChRatio), num_blocks=4)
        self.enc4 = EncoderStage(in_channels=int(64*ChRatio), out_channels=int(64*ChRatio), num_blocks=4)

        self.encdec = Conv2D(in_channels=int(64*ChRatio), out_channels=int(64*ChRatio), kernel_size=3, padding=1, stride=1, is_seperable=True, has_relu=True)
        self.dec1 = DecoderStage(in_channels=int(64*ChRatio), skip_in_channels=int(64*ChRatio), out_channels=int(64*ChRatio))
        self.dec2 = DecoderStage(in_channels=int(64*ChRatio), skip_in_channels=int(64*ChRatio), out_channels=int(32*ChRatio))
        self.dec3 = DecoderStage(in_channels=int(32*ChRatio), skip_in_channels=int(64*ChRatio), out_channels=int(32*ChRatio))
        self.dec4 = DecoderStage(in_channels=int(32*ChRatio), skip_in_channels=int(16), out_channels=16)

        self.out0 = DecoderBlock(in_channels=16, out_channels=16, kernel_size=3)
        self.out1 = Conv2D(in_channels=16, out_channels=4, kernel_size=3, stride=1, padding=1, is_seperable=False, has_relu=False)

class NetworkTimBrooks(NetworkPMRID):

    def __init__(self, ChRatio):
        super().__init__(ChRatio = ChRatio)

        self.conv0 = Conv2D(in_channels=8, out_channels=16, kernel_size=3, padding=1, stride=1, is_seperable=False, has_relu=True)

    def forward(self, inp):

        conv0 = self.conv0(inp)
        conv1 = self.enc1(conv0)
        conv2 = self.enc2(conv1)
        conv3 = self.enc3(conv2)
        conv4 = self.enc4(conv3)

        conv5 = self.encdec(conv4)

        up3 = self.dec1((conv5, conv3))
        up2 = self.dec2((up3, conv2))
        up1 = self.dec3((up2, conv1))
        x = self.dec4((up1, conv0))

        x = self.out0(x)
        x = self.out1(x)

        pred = inp[:, :4, :, :] + x
        return pred

class NetworkGolden4T(NetworkBasic):

    def __init__(self):
        super().__init__()

        self.conv0 = nn.Conv2d(8, 32, kernel_size=3, stride=1, padding=1, bias=True)
        self.act = nn.LeakyReLU()
        self.ds = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.us = nn.UpsamplingBilinear2d(scale_factor=2)

        self.Encoder1 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=True)
        self.Encoder2a = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=True)
        self.Encoder2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=True)
        self.Encoder3a = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=True)
        self.Encoder3 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=True)
        self.Encoder4a = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=True)
        self.Encoder4 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True)

        self.Encoder5a = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1, bias=True)
        self.Decoder5 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=True)

        self.Decoder4a = nn.Conv2d(512+256, 256, kernel_size=3, stride=1, padding=1, bias=True)
        self.Decoder4 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True)
        self.Decoder3a = nn.Conv2d(256+128, 128, kernel_size=3, stride=1, padding=1, bias=True)
        self.Decoder3 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=True)
        self.Decoder2a = nn.Conv2d(128+64, 64, kernel_size=3, stride=1, padding=1, bias=True)
        self.Decoder2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=True)
        self.Decoder1a = nn.Conv2d(64+32, 32, kernel_size=3, stride=1, padding=1, bias=True)
        self.Decoder1 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=True)

        self.convZ = nn.Conv2d(32, 4, kernel_size=3, stride=1, padding=1, bias=True)


    def forward(self, noisyConcat):
        conv0 = self.conv0(noisyConcat)

        Encoder1A = self.act(self.Encoder1(conv0))
        Encoder1B = self.act(self.Encoder1(Encoder1A))
        Encoder1C = self.act(self.Encoder1(Encoder1B))
        Encoder1D = self.act(self.Encoder1(Encoder1C))

        Encoder2A = self.act(self.Encoder2a(self.ds(Encoder1D)))
        Encoder2B = self.act(self.Encoder2(Encoder2A))
        Encoder2C = self.act(self.Encoder2(Encoder2B))
        Encoder2D = self.act(self.Encoder2(Encoder2C))

        Encoder3A = self.act(self.Encoder3a(self.ds(Encoder2D)))
        Encoder3B = self.act(self.Encoder3(Encoder3A))
        Encoder3C = self.act(self.Encoder3(Encoder3B))
        Encoder3D = self.act(self.Encoder3(Encoder3C))

        Encoder4A = self.act(self.Encoder4a(self.ds(Encoder3D)))
        Encoder4B = self.act(self.Encoder4(Encoder4A))
        Encoder4C = self.act(self.Encoder4(Encoder4B))
        Encoder4D = self.act(self.Encoder4(Encoder4C))

        Decoder5A = self.act(self.Encoder5a(self.ds(Encoder4D)))
        Decoder5B = self.act(self.Decoder5(Decoder5A))
        Decoder5C = self.act(self.Decoder5(Decoder5B))
        Decoder5D = self.act(self.Decoder5(Decoder5C))
        Decoder5Dus = torch.cat([Encoder4D, self.us(Decoder5D)], dim=1)

        Decoder4A = self.act(self.Decoder4a(Decoder5Dus))
        Decoder4B = self.act(self.Decoder4(Decoder4A))
        Decoder4C = self.act(self.Decoder4(Decoder4B))
        Decoder4D = self.act(self.Decoder4(Decoder4C))
        Decoder4Dus = torch.cat([Encoder3D, self.us(Decoder4D)], dim=1)

        Decoder3A = self.act(self.Decoder3a(Decoder4Dus))
        Decoder3B = self.act(self.Decoder3(Decoder3A))
        Decoder3C = self.act(self.Decoder3(Decoder3B))
        Decoder3D = self.act(self.Decoder3(Decoder3C))
        Decoder3Dus = torch.cat([Encoder2D, self.us(Decoder3D)], dim=1)

        Decoder2A = self.act(self.Decoder2a(Decoder3Dus))
        Decoder2B = self.act(self.Decoder2(Decoder2A))
        Decoder2C = self.act(self.Decoder2(Decoder2B))
        Decoder2D = self.act(self.Decoder2(Decoder2C))
        Decoder2Dus = torch.cat([Encoder1D, self.us(Decoder2D)], axis=1)

        Decoder1A = self.act(self.Decoder1a(Decoder2Dus))
        Decoder1B = self.act(self.Decoder1(Decoder1A))
        Decoder1C = self.act(self.Decoder1(Decoder1B))
        Decoder1D = self.act(self.Decoder1(Decoder1C))

        Pred = self.act(self.convZ(Decoder1D)) + noisyConcat[:, :4, :, :]

        return Pred

if __name__ == "__main__":
    # net, img = NetworkPMRID(ChRatio=0.5), torch.randn(1, 4, 64, 64, device=torch.device('cpu'), dtype=torch.float32)
    net, img = NetworkTimBrooks(ChRatio=0.5), torch.randn(1, 8, 64, 64, device=torch.device('cpu'), dtype=torch.float32)
    # net, img = NetworkGolden4T(), torch.randn(1, 8, 64, 64, device=torch.device('cpu'), dtype=torch.float32)
    # out = net(img)

    # summary(net, input_size=(1, 8, 64, 64))
    flops, params = profile(net, inputs=(img,))
    gflops = flops/1e9
    imgSizeM = img.shape[2]*img.shape[3]*4/1e6
    print(f"FLOPs: {gflops:.2f}G, Params: {params/1e6:.2f}M, gFlops per 1M pixel = {gflops/imgSizeM:.2f}G")