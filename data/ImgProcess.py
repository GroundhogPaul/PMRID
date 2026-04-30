import torch
import torch.nn.functional as F
from ImgUnprocess import unprocess

def apply_gains(bayer_images, red_gains, blue_gains):
    assert bayer_images.shape[-1] == 4
    dim = len(bayer_images.shape)
    assert dim == 3

    green_gains = torch.ones_like(red_gains)
    gains = torch.stack([red_gains, green_gains, green_gains, blue_gains], dim=-1)  # (batch, 4)
    if dim == 4:
        gains = gains[:, None, None, :]  # (batch, 1, 1, 4)
    return bayer_images * gains

def demosaic(bayer_images):
        # Convert from NHWC to NCHW for PyTorch ops
    bayer = bayer_images.permute(0, 3, 1, 2)  # (N, 4, H, W)
    N, _, H, W = bayer.shape
    new_H, new_W = H * 2, W * 2

    # Extract channels (keeping channel dim)
    red = bayer[:, 0:1, :, :]          # (N, 1, H, W)
    green_red = bayer[:, 1:2, :, :]    # (N, 1, H, W)
    green_blue = bayer[:, 2:3, :, :]   # (N, 1, H, W)
    blue = bayer[:, 3:4, :, :]         # (N, 1, H, W)

    # ----- Red channel -----
    # Resize bilinearly to 2x size
    red_up = F.interpolate(red, size=(new_H, new_W), mode='bilinear', align_corners=False)

    # ----- Green channel (complex) -----
    # Green from red positions
    g_red = torch.flip(green_red, dims=[3])          # horizontal flip
    g_red = F.interpolate(g_red, size=(new_H, new_W), mode='bilinear', align_corners=False)
    g_red = torch.flip(g_red, dims=[3])              # flip back
    g_red = torch.pixel_unshuffle(g_red, 2)          # (N, 4, H, W)

    # Green from blue positions
    g_blue = torch.flip(green_blue, dims=[2])        # vertical flip
    g_blue = F.interpolate(g_blue, size=(new_H, new_W), mode='bilinear', align_corners=False)
    g_blue = torch.flip(g_blue, dims=[2])            # flip back
    g_blue = torch.pixel_unshuffle(g_blue, 2)        # (N, 4, H, W)

    # Combine the two green estimates
    green_at_red = (g_red[:, 0, :, :] + g_blue[:, 0, :, :]) / 2.0
    green_at_green_red = g_red[:, 1, :, :]
    green_at_green_blue = g_blue[:, 2, :, :]
    green_at_blue = (g_red[:, 3, :, :] + g_blue[:, 3, :, :]) / 2.0

    # Stack the four green planes and apply pixel_shuffle to reconstruct full green
    green_planes = torch.stack([
        green_at_red, green_at_green_red, green_at_green_blue, green_at_blue
    ], dim=1)  # (N, 4, H, W)
    green = torch.pixel_shuffle(green_planes, 2)    # (N, 1, 2H, 2W)

    # ----- Blue channel -----
    # Flip both horizontally and vertically, then upscale, then flip back
    blue_flipped = torch.flip(blue, dims=[2])        # vertical
    blue_flipped = torch.flip(blue_flipped, dims=[3]) # horizontal
    blue_up = F.interpolate(blue_flipped, size=(new_H, new_W), mode='bilinear', align_corners=False)
    blue_up = torch.flip(blue_up, dims=[3])          # horizontal back
    blue_up = torch.flip(blue_up, dims=[2])          # vertical back

    # ----- Concatenate RGB -----
    rgb = torch.cat([red_up, green, blue_up], dim=1)  # (N, 3, 2H, 2W)

    # Back to NHWC format
    rgb = rgb.permute(0, 2, 3, 1)                    # (N, 2H, 2W, 3)
    return rgb

def apply_ccms(images, ccms):
    """Applies color correction matrices."""
    # 确保输入为四维张量 (batch, height, width, channels)
    assert images.dim() == 4, "images must be 4-dimensional"
    
    # 扩展维度以支持广播
    # images: (N, H, W, C) -> (N, H, W, 1, C)
    images = images.unsqueeze(-2)
    # ccms: (N, 3, 3) -> (N, 1, 1, 3, 3)
    ccms = ccms[:, None, None, :, :]  # 等价于 unsqueeze(1).unsqueeze(2)
    
    # 逐元素乘后对最后一个维度求和
    return (images * ccms).sum(dim=-1)

def gamma_compression(images, gamma=2.2):
    """Converts from linear to gamma space."""
    # Clamps to prevent numerical instability of gradients near zero.
    clamped = torch.clamp(images, min=1e-8)   # 等价于 tf.maximum(images, 1e-8)
    return clamped ** (1.0 / gamma)

def process(bayer_images, cam2rgbs, safe_gains):
    assert isinstance(bayer_images, torch.Tensor), "Input must be a torch tensor"
    assert bayer_images.shape[-1] == 4
    dim = bayer_images.dim()
    assert dim == 3 or dim == 4
    assert cam2rgbs.dim() == (dim - 1)
    assert safe_gains.dim() == dim

    if dim == 3: # must add batch dimension to run 'torch.pixel_shuffle' and 'F.interpolate'
        bayer_images = bayer_images[None, :, :, :]  # (1, H, W, 4)
        cam2rgbs = cam2rgbs[None, :, :]  # (1, 3, 3)
        safe_gains = safe_gains[None, :]   # (1, 4)

    # Demosaic.
    bayer_images = torch.clamp(bayer_images, 0.0, 1.0)
    images = demosaic(bayer_images)
    # White balance.
    images = images / safe_gains
    # Color correction.
    images = apply_ccms(images, cam2rgbs)
    # Gamma compression.
    images = torch.clamp(images, 0.0, 1.0)
    images = gamma_compression(images)

    # rgb to bgr for opencv
    images = images[..., [2, 1, 0]]  # (N, H, W, 3)

    if dim == 3:
        images = images[0]  # (H, W, 3)

    images = (images * 255.0).byte()  # (N, H, W, 3) or (H, W, 3)
    return images

def process_meta_data(bayer_images, meta_data):
    assert isinstance(bayer_images, torch.Tensor)
    cam2rgbs = meta_data['ccm3x3']
    safe_gains = meta_data['safe_gains']

    return process(bayer_images, cam2rgbs, safe_gains)

if __name__ == '__main__':
    import cv2

    img_path = 'im1.jpg'
    rgb888 = cv2.imread(img_path)
    bgr888 = cv2.cvtColor(rgb888, cv2.COLOR_BGR2RGB)
    device = 'cpu'
    bgr888 = torch.from_numpy(bgr888).to(device)

    HWCh4n, meta_data = unprocess(bgr888)
    bgr888out = process_meta_data(HWCh4n, meta_data)
    bgr888out = bgr888out.cpu().numpy()  # (H, W, 3)
    cv2.imwrite("img1_out.jpg", bgr888out)