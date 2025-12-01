import utilBasicRun
from models.net_torch import *
import matplotlib.pyplot as plt

import numpy as np
import os
import cv2
from utilArw import ArwReader
from utilRaw import RawUtils
import pickle

from run_benchmark import KSigma

# ---------- collect one ISO dngs ---------- #
sPathArw = r"00002_00_10s.ARW"
# sPathArw = r"00012_00_10s.ARW"
ArwReaderObj = ArwReader(sPathArw)

bayer_visible = ArwReaderObj.raw_image_visible.astype(np.float32) if ArwReaderObj.raw_image_visible is not None else None
bayer_01 = bayer_visible / ArwReaderObj.white_level
rggb_01 = RawUtils.bayer2rggb(bayer_01)

ISO = 12800
KSigmaObj = KSigma(
    K_coeff=[0.0005995267, 0.00868861],
    B_coeff=[7.11772e-7, 6.514934e-4, 0.11492713],
    anchor=1600,
)

k, sigma = KSigmaObj.GetKSigma(ISO)
bayer_10bit = bayer_01 * 1024 # add noise on 10bit because the k and sigma are caliberated on 10 bit
print("ISO =", ISO, " k =", k, " sigma =", sigma)
shot_noise = torch.poisson(torch.from_numpy(bayer_10bit) / k) * k #
read_noise = torch.randn(bayer_10bit.shape) * torch.sqrt(torch.tensor(sigma))
bayer_noisy_10bit = np.clip(shot_noise + read_noise, 0, 1024)

bayer_noisy_01 = bayer_noisy_10bit / 1024
bayer_noisy_01 = bayer_noisy_01.numpy()

rgb_noisy_01 = RawUtils.bayer01_2_rgb01(bayer_noisy_01, wb_gain=[1.5156, 1.0, 1.7421], CCM=np.eye(3))
rgb_01 = RawUtils.bayer01_2_rgb01(bayer_01, wb_gain=[1.5156, 1.0, 1.7421], CCM=np.eye(3))
cv2.imwrite("rgb_noisy.bmp", (rgb_noisy_01*255.0).astype(np.uint8))
cv2.imwrite("rgb_clean.bmp", (rgb_01*255.0).astype(np.uint8))

rgb_noisy = rgb_noisy_01 * 255
rgb = rgb_01 * 255

import skimage
# print(rgb.max(), rgb.min())
psnr_rgb = skimage.metrics.peak_signal_noise_ratio(rgb.astype(np.uint8), rgb_noisy.astype(np.uint8))
# print(bayer_01.min(), bayer_01.max())
# print(bayer_noisy_01.min(), bayer_noisy_01.max())
psnr_bayer = skimage.metrics.peak_signal_noise_ratio(bayer_01, bayer_noisy_01, data_range = 1.0)
print("psnr_bayer01 = ", psnr_bayer, ", psnr_rgb = ", psnr_rgb)