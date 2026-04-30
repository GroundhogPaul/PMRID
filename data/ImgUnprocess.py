import torch

def InverseSmoothStep_GammaExpansion_LUT(image):
    lut_list = [
        2.51187993e-18, 6.91470923e-04, 1.49959628e-03, 2.36394047e-03,
        3.26936133e-03, 4.20818478e-03, 5.17569575e-03, 6.16866397e-03,
        7.18473224e-03, 8.22210032e-03, 9.27932188e-03, 1.03552481e-02,
        1.14489412e-02, 1.25595517e-02, 1.36864176e-02, 1.48289185e-02,
        1.59865487e-02, 1.71588641e-02, 1.83454342e-02, 1.95459258e-02,
        2.07599979e-02, 2.19873749e-02, 2.32277829e-02, 2.44809855e-02,
        2.57467907e-02, 2.70249825e-02, 2.83153895e-02, 2.96178497e-02,
        3.09322290e-02, 3.22583579e-02, 3.35961357e-02, 3.49454507e-02,
        3.63061838e-02, 3.76782455e-02, 3.90615426e-02, 4.04560044e-02,
        4.18615490e-02, 4.32781056e-02, 4.47056331e-02, 4.61440273e-02,
        4.75932881e-02, 4.90533337e-02, 5.05241565e-02, 5.20056896e-02,
        5.34979068e-02, 5.50007634e-02, 5.65142445e-02, 5.80383353e-02,
        5.95730282e-02, 6.11182824e-02, 6.26740977e-02, 6.42404333e-02,
        6.58173189e-02, 6.74047321e-02, 6.90026730e-02, 7.06111491e-02,
        7.22301453e-02, 7.38596544e-02, 7.54997209e-02, 7.71503299e-02,
        7.88115039e-02, 8.04832652e-02, 8.21656063e-02, 8.38585347e-02,
        8.55620950e-02, 8.72763097e-02, 8.90012011e-02, 9.07367468e-02,
        9.24830288e-02, 9.42400694e-02, 9.60078686e-02, 9.77864936e-02,
        9.95759368e-02, 1.01376280e-01, 1.03187509e-01, 1.05009697e-01,
        1.06842890e-01, 1.08687080e-01, 1.10542372e-01, 1.12408765e-01,
        1.14286356e-01, 1.16175123e-01, 1.18075214e-01, 1.19986564e-01,
        1.21909313e-01, 1.23843469e-01, 1.25789106e-01, 1.27746314e-01,
        1.29715100e-01, 1.31695539e-01, 1.33687705e-01, 1.35691658e-01,
        1.37707472e-01, 1.39735207e-01, 1.41774938e-01, 1.43826738e-01,
        1.45890683e-01, 1.47966847e-01, 1.50055259e-01, 1.52156085e-01,
        1.54269338e-01, 1.56395137e-01, 1.58533573e-01, 1.60684720e-01,
        1.62848637e-01, 1.65025443e-01, 1.67215243e-01, 1.69418111e-01,
        1.71634138e-01, 1.73863441e-01, 1.76106140e-01, 1.78362325e-01,
        1.80632055e-01, 1.82915494e-01, 1.85212716e-01, 1.87523872e-01,
        1.89849064e-01, 1.92188412e-01, 1.94541991e-01, 1.96909979e-01,
        1.99292496e-01, 2.01689646e-01, 2.04101622e-01, 2.06528440e-01,
        2.08970383e-01, 2.11427480e-01, 2.13899881e-01, 2.16387808e-01,
        2.18891367e-01, 2.21410751e-01, 2.23946035e-01, 2.26497352e-01,
        2.29065031e-01, 2.31649145e-01, 2.34249771e-01, 2.36867249e-01,
        2.39501685e-01, 2.42153287e-01, 2.44822145e-01, 2.47508556e-01,
        2.50212729e-01, 2.52934843e-01, 2.55674988e-01, 2.58433521e-01,
        2.61210591e-01, 2.64006436e-01, 2.66821235e-01, 2.69655347e-01,
        2.72508860e-01, 2.75382161e-01, 2.78275371e-01, 2.81188816e-01,
        2.84122616e-01, 2.87077308e-01, 2.90053010e-01, 2.93049902e-01,
        2.96068460e-01, 2.99108922e-01, 3.02171528e-01, 3.05256695e-01,
        3.08364660e-01, 3.11495751e-01, 3.14650327e-01, 3.17828834e-01,
        3.21031511e-01, 3.24258715e-01, 3.27510893e-01, 3.30788404e-01,
        3.34091693e-01, 3.37421209e-01, 3.40777189e-01, 3.44160229e-01,
        3.47570717e-01, 3.51009220e-01, 3.54476035e-01, 3.57971847e-01,
        3.61497015e-01, 3.65052164e-01, 3.68637711e-01, 3.72254431e-01,
        3.75902861e-01, 3.79583359e-01, 3.83296818e-01, 3.87043744e-01,
        3.90824884e-01, 3.94640833e-01, 3.98492336e-01, 4.02380198e-01,
        4.06305134e-01, 4.10267949e-01, 4.14269567e-01, 4.18310493e-01,
        4.22392040e-01, 4.26514924e-01, 4.30680275e-01, 4.34888899e-01,
        4.39141870e-01, 4.43440408e-01, 4.47785586e-01, 4.52178597e-01,
        4.56620663e-01, 4.61113095e-01, 4.65657383e-01, 4.70254719e-01,
        4.74906743e-01, 4.79615062e-01, 4.84381318e-01, 4.89207089e-01,
        4.94094372e-01, 4.99044985e-01, 5.04061043e-01, 5.09144604e-01,
        5.14297962e-01, 5.19523621e-01, 5.24824083e-01, 5.30201852e-01,
        5.35659850e-01, 5.41201353e-01, 5.46829462e-01, 5.52547634e-01,
        5.58359623e-01, 5.64269483e-01, 5.70281327e-01, 5.76399863e-01,
        5.82630038e-01, 5.88977337e-01, 5.95447361e-01, 6.02046669e-01,
        6.08782053e-01, 6.15661144e-01, 6.22692049e-01, 6.29884064e-01,
        6.37247145e-01, 6.44792676e-01, 6.52532876e-01, 6.60481811e-01,
        6.68655336e-01, 6.77071214e-01, 6.85749650e-01, 6.94714487e-01,
        7.03992367e-01, 7.13615179e-01, 7.23620653e-01, 7.34053314e-01,
        7.44967878e-01, 7.56431341e-01, 7.68527985e-01, 7.81366944e-01,
        7.95091629e-01, 8.09899986e-01, 8.26076269e-01, 8.44057143e-01,
        8.64577770e-01, 8.89070630e-01, 9.21234548e-01, 1.00000000e+00]

    LUTtable = torch.tensor(lut_list, dtype=torch.float32, device=image.device)
    return LUTtable[image.long()]

def random_ccm(device):
    """Generates random RGB -> Camera color correction matrices."""
    # Takes a random convex combination of XYZ -> Camera CCMs.
    xyz2cams = [[[1.0234, -0.2969, -0.2266],
                [-0.5625, 1.6328, -0.0469],
                [-0.0703, 0.2188, 0.6406]],
                [[0.4913, -0.0541, -0.0202],
                [-0.613, 1.3513, 0.2906],
                [-0.1564, 0.2151, 0.7183]],
                [[0.838, -0.263, -0.0639],
                [-0.2887, 1.0725, 0.2496],
                [-0.0627, 0.1427, 0.5438]],
                [[0.6596, -0.2079, -0.0562],
                [-0.4782, 1.3016, 0.1933],
                [-0.097, 0.1581, 0.5181]]]
    xyz2cams = torch.tensor(xyz2cams, dtype=torch.float32, device=device)  # (4, 3, 3)
    num_ccms = xyz2cams.shape[0]

    # Random convex combination weights (uniform in [1e-8, 1e8])
    weights = torch.empty(num_ccms, 1, 1, device=device).uniform_(1e-8, 1e8)   # (4, 1, 1)
    weights_sum = weights.sum(dim=0)                                            # (1, 1)
    xyz2cam = (xyz2cams * weights).sum(dim=0) / weights_sum                     # (3, 3)

    # RGB -> XYZ matrix (sRGB to XYZ)
    rgb2xyz = torch.tensor([[0.4124564, 0.3575761, 0.1804375],
                            [0.2126729, 0.7151522, 0.0721750],
                            [0.0193339, 0.1191920, 0.9503041]], dtype=torch.float32, device=device)

    # RGB -> Camera matrix
    rgb2cam = xyz2cam @ rgb2xyz   # equivalent to torch.mm(xyz2cam, rgb2xyz)

    # Row normalization (each row sums to 1)
    rgb2cam = rgb2cam / rgb2cam.sum(dim=-1, keepdim=True)

    # rgb2cam = torch.tensor([[1,0,0],[0,1,0],[0,0,1]], dtype = torch.float32, device=device)

    return rgb2cam

def random_gains(device):
    """Generates random gains for brightening and white balance."""
    # Red and blue gains represent white balance.
    rgb_gain = torch.tensor(1.0) / torch.normal(mean=0.8, std=0.1, size=(1,), device=device)
    red_gain = torch.empty(1, device=device).uniform_(1.9, 2.4)
    blue_gain = torch.empty(1, device=device).uniform_(1.5, 1.9)
    return rgb_gain, red_gain, blue_gain

def inverse_smoothstep(image):
    """Approximately inverts a global tone mapping curve."""
    # image = torch.clamp(image, 0.0, 1.0)
    # return 0.5 - torch.sin(torch.asin(1.0 - 2.0 * image) / 3.0)
    return image

def gamma_expansion(image):
    return torch.clamp(image, min=1e-8) ** 2.2

def apply_ccm(image, ccm):
    original_shape = image.shape
    flat_image = image.reshape(-1, 3)
    transformed = flat_image @ ccm.T
    return transformed.reshape(original_shape)

def safe_invert_gains(image, rgb_gain, red_gain, blue_gain):
    """Inverts gains while safely handling saturated pixels."""
    # 计算增益向量并调整维度至 [1, 1, 3]
    one = torch.tensor([1.0], device=red_gain.device, dtype=red_gain.dtype)
    gains = torch.stack([one / red_gain, one, one / blue_gain]) / rgb_gain
    gains = gains.view(1, 1, -1)   # [1, 1, 3]

    # 图像灰度平均值（保持最后一维）
    gray = image.mean(dim=-1, keepdim=True)  # [..., 1]

    inflection = 0.9
    # 平滑掩码: (max(gray - 0.9, 0) / 0.1)^2
    mask = (torch.clamp(gray - inflection, min=0.0) / (1.0 - inflection)) ** 2.0

    # 安全增益：在最大值处保护饱和像素
    safe_gains = torch.maximum(mask + (1.0 - mask) * gains, gains)

    return image * safe_gains, safe_gains

def mosaic(image):
    """Extracts RGGB Bayer planes from an RGB image."""
    # image.shape.assert_is_compatible_with((None, None, 3))
    shape = image.shape
    red = image[0::2, 0::2, 0]
    green_red = image[0::2, 1::2, 1]
    green_blue = image[1::2, 0::2, 1]
    blue = image[1::2, 1::2, 2]

    # 沿最后一维堆叠，然后重塑为 (H/2, W/2, 4)
    stacked = torch.stack([red, green_red, green_blue, blue], dim=-1)
    # 原始形状推断需要，但直接用 shape 可能不可导，使用 .shape 属性
    H, W = shape[0], shape[1]
    return stacked.reshape(H // 2, W // 2, 4)

def unprocess(bgr888):
    assert isinstance(bgr888, torch.Tensor), "Input must be a torch tensor"
    assert len(bgr888.shape) == 3
    assert bgr888.shape[2] == 3

    # Randomly creates image metadata.
    ccm = random_ccm(bgr888.device)
    rgb_gain, red_gain, blue_gain = random_gains(bgr888.device)

    # ---------- if use LUT ---------- #
    HWCh3n = InverseSmoothStep_GammaExpansion_LUT(bgr888)
    # ---------- if not use LUT ---------- #
    # Approximately inverts global tone mapping.
    # image = image / 255.0
    # image = inverse_smoothstep(image)
    # Inverts gamma compression.
    # image = gamma_expansion(image)
    # Inverts color correction.
    # ccm = torch.tensor([[1,0,0],[0,1,0],[0,0,1]], dtype=image.dtype)
    # ---------- LUT end ---------- #

    HWCh3n = apply_ccm(HWCh3n, ccm)
    # Approximately inverts white balance and brightening.
    HWCh3n, safe_gains = safe_invert_gains(HWCh3n, rgb_gain, red_gain, blue_gain)
    # # Clips saturated pixels.
    HWCh3n = torch.clamp(HWCh3n, 0.0, 1.0)
    # # Applies a Bayer mosaic.
    HWCh4n = mosaic(HWCh3n)

    metadata = {
        'ccm3x3': torch.inverse(ccm),
        'safe_gains': safe_gains, # the really used "WB gain * DGain" is here
    }
    return HWCh4n, metadata