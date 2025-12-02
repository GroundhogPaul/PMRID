import utilBasicRun

import numpy as np
import skimage
import os
import cv2
from utilRaw import RawUtils
from run_benchmark import KSigma, Denoiser
from utilVrf import read_vrf, save_vrf_image, save_raw_image

# ---------- read vrf ---------- #
# ----- assert paths ----- #
sFolder = r"D:\image_database\jn1_mfnr_bestshot\unpacked"
assert os.path.exists(sFolder), f"Data folder does not exist: {sFolder}"
sVrfFile = "53/0_unpacked.vrf"
# sVrfFile = "33/5_unpacked.vrf"
sVrfPath = os.path.join(sFolder, sVrfFile)
assert os.path.exists(sVrfPath), f"VRF file does not exist: {sVrfPath}"

# ----- read vrf ----- #
W, H = 4080, 3060
black_level = 64
white_level = 1023
dgain = 1.0

bayer01_GRBG_noisy = read_vrf(sVrfPath, W, H, black_level, dgain, white_level)
bayer01_RGGB_noisy = np.fliplr(bayer01_GRBG_noisy)
bayer01_BGGR_noisy = RawUtils.rggb2bggr(bayer01_RGGB_noisy)

bgr01_noisy = RawUtils.bayer01_2_rgb01(bayer01_BGGR_noisy, wb_gain=[1.5156, 1.0, 1.7421], CCM=np.eye(3))
bgr_noisy = (bgr01_noisy*255.0).astype(np.uint8)
cv2.imwrite("noisy_rgb.png", bgr_noisy)

# ---------- Denoise ---------- #
kSigma = KSigma(
    K_coeff=[0.0005995267, 0.00868861],
    B_coeff=[7.11772e-7, 6.514934e-4, 0.11492713],
    anchor=1600,
)

model_path =  "D:/users/xiaoyaopan/PxyAI/PMRID_OFFICIAL/PMRID/models/torch_pretrained.ckp"
Denoiser = Denoiser(model_path, kSigma)

bayer01_RGGB_denoise = Denoiser.run(bayer01_RGGB_noisy, iso=6400.0)
bayer01_RGGB_denoise = np.clip(bayer01_RGGB_denoise, 0.0, 1.0)

# ---------- save image ---------- #
# ----- save png ----- #
bayer01_BGGR_denoise = RawUtils.rggb2bggr(bayer01_RGGB_denoise)

bgr01_denoise = RawUtils.bayer01_2_rgb01(bayer01_BGGR_denoise, wb_gain=[1.5156, 1.0, 1.7421], CCM=np.eye(3))
bgr_denoise = (bgr01_denoise*255.0).astype(np.uint8)
cv2.imwrite("denoise_rgb.bmp", bgr_denoise)

import skimage
psnr = skimage.metrics.peak_signal_noise_ratio(bgr_denoise, bgr_noisy)
print("psnr_bgr = ", psnr)
print(bayer01_RGGB_denoise.min(), bayer01_RGGB_denoise.max())
print(bayer01_RGGB_noisy.min(), bayer01_RGGB_noisy.max())
psnr = skimage.metrics.peak_signal_noise_ratio(bayer01_RGGB_denoise, bayer01_RGGB_noisy) 
print("psnr_bayer01 = ", psnr)

bgr_denoise_std = cv2.imread("denoise_rgb_std.bmp")
errNorm2 = np.linalg.norm(bgr_denoise.astype(np.float32) - bgr_denoise_std.astype(np.float32))
print("errNorm2 = ", errNorm2)

# ----- save vrf ----- #
out_ratio = 4  #out 12bit
out_black_level = black_level * out_ratio  # 根据实际情况调整
out_white_level = (white_level + 1) * out_ratio - 1
bayer01_GRBG_denoise = np.fliplr(bayer01_RGGB_denoise)
denoised_image = save_raw_image(bayer01_GRBG_denoise, os.path.join("", '53.raw'), out_white_level, out_black_level)
save_vrf_image(denoised_image, sVrfPath, os.path.join("", '53.vrf'), out_white_level)