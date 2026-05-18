import torch
import torch.nn as nn
import torch.nn.functional as F

from thop import profile
from torchinfo import summary


class ResidualBlock(nn.Module):
    """Basic residual block"""
    def __init__(self, num_filters):
        super(ResidualBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(num_filters, num_filters, kernel_size=3, stride=1, padding=1, padding_mode='zeros', groups = 2)
        self.conv2 = nn.Conv2d(num_filters, num_filters, kernel_size=1, stride=1, padding=0, padding_mode='zeros')
        self.relu = nn.LeakyReLU(inplace=True)
        
    def forward(self, x):
        identity = x
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))    
        return x + identity


class EncoderBlock(nn.Module):
    """Encoder block: residual block with downsampling"""
    def __init__(self, in_channels, out_channels, stride=2):
        super(EncoderBlock, self).__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, padding_mode='zeros')
        self.relu = nn.LeakyReLU(inplace=True)
        self.residual_block = ResidualBlock(out_channels)
    
    def forward(self, x):
        x = self.relu(self.conv(x))
        x = self.residual_block(x)
        return x


class DecoderBlock(nn.Module):
    """Decoder block: upsample, concatenate, then residual block"""
    def __init__(self, num_filters):
        super(DecoderBlock, self).__init__()
        
        self.deconv = nn.ConvTranspose2d(num_filters, num_filters, kernel_size=2, stride=2, padding=0)
        self.conv = nn.Conv2d(num_filters*2, num_filters, kernel_size=3, stride=1, padding=1, padding_mode='zeros')
        self.residual_block = ResidualBlock(num_filters)

    def forward(self, x, skip_connection):
        x = self.deconv(x)
        x = torch.cat([skip_connection, x], dim=1)
        x = self.conv(x)
        x = self.residual_block(x)
        return x


class NOAHTCVgroup(nn.Module):
    """
    Denoising U-Net with simplified architecture using residual blocks.
    """
    def __init__(self, num_filters=32):
        super(NOAHTCVgroup, self).__init__()
        self.num_filters = num_filters
        
        # Encoder 
        self.enc0 = EncoderBlock(          8, num_filters, stride=1)      # full resolution
        self.enc1 = EncoderBlock(num_filters, num_filters, stride=2)      # full -> 1/2 resolution
        self.enc2 = EncoderBlock(num_filters, num_filters, stride=2)      # 1/2  -> 1/4 resolution

        # Decoder 
        self.dec2 = DecoderBlock(num_filters)   # 1/4 -> 1/2 resolution
        self.dec1 = DecoderBlock(num_filters)   # 1/2 -> full resolution

        # Final layers
        self.final_conv = nn.Sequential(
            nn.Conv2d(num_filters, num_filters, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(num_filters, 4, kernel_size=3, stride=1, padding=1)  # Output residual
        )
        
    def forward(self, noisy_img, std_map):
        """
        Args:
            noisy_img: Tensor of shape [B, 4, H, W] (Bayer planes)
            std_map: Tensor of shape [B, 4, H, W] (Noise variance)
        
        Returns:
            denoised_img: Tensor of shape [B, 4, H, W]
        """
        # Concatenate input
        features = torch.cat([noisy_img, std_map], dim=1)  # [B, 8, H, W]
        
        # Encoder
        skip1 = self.enc0(features)    # [B, num_filters, H, W]
        skip2 = self.enc1(skip1)       # [B, num_filters, H/2, W/2]
        x = self.enc2(skip2)           # [B, num_filters, H/4, W/4]
        
        
        # Decoder with skip connections
        x = self.dec2(x, skip2)        # [B, num_filters, H/2, W/2]
        x = self.dec1(x, skip1)        # [B, num_filters, H, W]
        
        # Final output
        residual = self.final_conv(x)   # [B, 4, H, W]
        denoised_img = noisy_img + residual
        
        return denoised_img

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

        self.load_state_dict(state_dict)
        print(f"Loaded model weights from {sCKPT}.")
    

class NOAHTCVgroup_Level3(nn.Module):
    """
    Denoising U-Net with simplified architecture using residual blocks.
    """
    def __init__(self, num_filters=32):
        super(NOAHTCVgroup_Level3, self).__init__()
        self.num_filters = num_filters
        
        # Encoder 
        self.enc0 = EncoderBlock(          8, num_filters, stride=1)      # full resolution
        self.enc1 = EncoderBlock(num_filters, num_filters, stride=2)      # full -> 1/2 resolution
        self.enc2 = EncoderBlock(num_filters, num_filters, stride=2)      # 1/2  -> 1/4 resolution
        self.enc3 = EncoderBlock(num_filters, num_filters, stride=2)      # 1/4  -> 1/8 resolution

        # Decoder 
        self.dec3 = DecoderBlock(num_filters)   # 1/8 -> 1/4 resolution
        self.dec2 = DecoderBlock(num_filters)   # 1/4 -> 1/2 resolution
        self.dec1 = DecoderBlock(num_filters)   # 1/2 -> full resolution

        # Final layers
        self.final_conv = nn.Sequential(
            nn.Conv2d(num_filters, num_filters, kernel_size=3, stride=1, padding=1, groups = 2),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(num_filters, 4, kernel_size=3, stride=1, padding=1)  # Output residual
        )
        
    def forward(self, noisy_img, std_map):
        """
        Args:
            noisy_img: Tensor of shape [B, 4, H, W] (Bayer planes)
            std_map: Tensor of shape [B, 4, H, W] (Noise variance)
        
        Returns:
            denoised_img: Tensor of shape [B, 4, H, W]
        """
        # Concatenate input
        features = torch.cat([noisy_img, std_map], dim=1)  # [B, 8, H, W]
        
        # Encoder
        skip1 = self.enc0(features)    # [B, num_filters, H, W]
        skip2 = self.enc1(skip1)       # [B, num_filters, H/2, W/2]
        skip3 = self.enc2(skip2)       # [B, num_filters, H/4, W/4]
        x = self.enc3(skip3)           # [B, num_filters, H/8, W/8]
        
        
        # Decoder with skip connections
        x = self.dec3(x, skip3)        # [B, num_filters, H/4, W/4]
        x = self.dec2(x, skip2)        # [B, num_filters, H/2, W/2]
        x = self.dec1(x, skip1)        # [B, num_filters, H, W]
        
        # Final output
        residual = self.final_conv(x)   # [B, 4, H, W]
        denoised_img = noisy_img + residual
        
        return denoised_img

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

        self.load_state_dict(state_dict)
        self.to(device)
        print(f"Loaded model weights from {sCKPT}.")

if __name__ == '__main__':
    # ---------- test 1: train mode run and cal GFLops ---------- #
    H, W, Ch = 512, 512, 4
    device = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')
    net = NOAHTCVgroup_Level3().to(device)

    img, var = torch.randn(1, 4, H, W, device=device, dtype=torch.float32), torch.randn(1, Ch, H, W, device=device, dtype=torch.float32)

    out = net(img, var)
    summary(net, input_size=[(1, 4, H, W), (1, 4, H, W)])
    flops, params = profile(net, inputs=(img, var))

    print(f"FLOPs Total: {flops/1e9:.2f}G")
    print(f"FLOPs/1M: {flops/1e9/(H*W*4/1024/1024):.2f}G")
    print(f"Params: {params/1e6:.2f}M")