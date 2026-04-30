import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)
import utilData

import torch
import copy
from utils.KSigma import KSigma, Official_Ksigma_params
from ImgDataset import CropDatasetJpg

def add_noise_Concat(ChHW_01):
    assert isinstance(ChHW_01, torch.Tensor), "Input should be a PyTorch tensor"
    assert ChHW_01.dim() == 3, "Input should have 3 dimensions [C, H, W]"
    # if not [C,H,W], an extra permute has to be done on ChHW_noise_var
    assert ChHW_01.shape[0] == 4, "Input should have 4 channels (C=4), representing RGGB"

    # ----- Case 1: Tim Brooks paper ----- #
    log_min_shot_noise = torch.log(torch.tensor(0.0001))
    log_max_shot_noise = torch.log(torch.tensor(0.012))
    log_shot_noise =  log_min_shot_noise + (log_max_shot_noise - log_min_shot_noise) * torch.rand(1)
    shot_noise = torch.exp(log_shot_noise).item()

    line = lambda x: 2.18 * x + 1.2
    log_read_noise = line(log_shot_noise) + torch.normal(mean = 0.0, std = 0.26, size=())
    read_noise = torch.exp(log_read_noise).item()

    # ----- Case 2: Jin1 LuoWen Calibration ----- #
    # log_min_shot_noise = torch.log(torch.tensor(0.0013)) # 64x AGain, Jin1 shot_noise = 0.00235 
    # log_max_shot_noise = torch.log(torch.tensor(0.0033))
    # log_shot_noise =  log_min_shot_noise + (log_max_shot_noise - log_min_shot_noise) * torch.rand(1)
    # shot_noise = torch.exp(log_shot_noise).item()

    # log_min_read_noise = torch.log(torch.tensor(5e-6)) # 64x AGain, Jin1 read_noise = -1.11e-5 
    # log_max_read_noise = torch.log(torch.tensor(2e-5))
    # log_read_noise =  log_min_read_noise + (log_max_read_noise - log_min_read_noise) * torch.rand(1)
    # read_noise = torch.exp(log_read_noise).item()

    # ----- Case 3: Jin1 LuoWen 64x Gain
    # shot_noise = 0.00251
    # read_noise = 1.265e-5

    # ----- noise array ----- #
    variance = ChHW_01 * shot_noise + read_noise
    noise_01 = torch.randn_like(ChHW_01) * torch.sqrt(variance)
    ChHW_noisy = ChHW_01 + noise_01
    ChHW_noisy = torch.clamp(ChHW_noisy, 0, 1)

    ChHW_noise_var = torch.sqrt(ChHW_noisy * shot_noise + read_noise)
        
    return ChHW_noisy, ChHW_noise_var

def add_noise_KSigma(input_bayer_01, noise_type='Gaussian', blcClip = 0):
    iso = torch.randint(100, 9600, (1,))
    # KSigmaCur = KSigma(Official_Ksigma_params['K_coeff'], Official_Ksigma_params['B_coeff'], Official_Ksigma_params['anchor'])
    KSigmaCur = KSigma(
        Official_Ksigma_params['K_coeff'], 
        Official_Ksigma_params['B_coeff'],
        Official_Ksigma_params['anchor'], 
        # k = 0.00251* 1023, # LuoWen param under ISO 6400
        # sigma = 1.265e-05 * 1023 * 1023
        )

    iso = iso.item()
    k, sigma = KSigmaCur.GetKSigma(iso)
    kSigmaCalibLevel = 959 # 1023 - black_level(64), copy from run_benchmark.py
    input_bayer = input_bayer_01 * kSigmaCalibLevel # to kSigma calibrate scale 

    if noise_type == 'PoissonGaussian':
        # Poisson and Gaussian noise model
        shot_noise = torch.poisson(input_bayer / k) * k
        read_noise = torch.randn(input_bayer.shape) * torch.sqrt(torch.tensor(sigma))
        noisy_bayer = shot_noise + read_noise
        
    elif noise_type == 'Gaussian':
        # Gaussian noise model
        input_bayer = input_bayer
        variance = input_bayer * k + sigma
        noise = torch.randn_like(input_bayer) * torch.sqrt(variance)
        noisy_bayer = input_bayer + noise

    else:
        noisy_bayer = input_bayer

    noisy_bayer_01 = noisy_bayer / kSigmaCalibLevel  # to original scale

    return torch.clamp(noisy_bayer_01.to(torch.float32), blcClip, 1), copy.deepcopy(KSigmaCur)

if __name__ == "__main__":
    import numpy as np
    import cv2

    seed = 39
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')

    # --------- test jpg ---------- #
    dir_pattern = "D:/image_database/mirflickr25k/mirflickr/*.jpg"
    dataset = CropDatasetJpg(dir_pattern, Hbayer=1024, Wbayer=1024, device=device)
    input_rggb, meta_data = dataset[1]
    noisy_rggb, shot_noise, read_noise = add_noise_Concat(input_rggb)

    bgr888 = dataset.rggb01_2_bgr888(noisy_rggb, meta_data, bCPU = True)
    cv2.imwrite("test_AddNoiseConcat.jpg", bgr888)