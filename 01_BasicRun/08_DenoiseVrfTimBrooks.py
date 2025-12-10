import utilBasicRun

import numpy as np
import skimage
import os
import cv2
from utilRaw import RawUtils
from run_benchmark import KSigma, Official_Ksigma_params
from utilVrf import vrf, read_vrf, save_vrf_image, save_raw_image 
from models.net_torch import NetworkTimBrooks as Network
import torch
import shutil
import glob

# ---------- read model ---------- #
# ----- assert ckpt paths ----- #
model_path =  "./models/TIM_BROOKS_test/top_models/top_model_psnr_50.32_step_306000.pth"
assert os.path.exists(model_path), f"Model file does not exist: {model_path}"

# ----- load ckpt ----- #
device = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')
net = Network().to(device)
net.load_CKPT(str(model_path), device=torch.device(device))
net.eval()

# ----- output folder ----- #
sModel_folder = os.path.dirname(os.path.dirname(model_path))
sOut_folder = os.path.join(sModel_folder, 'denoise_vrf_out')
os.makedirs(sOut_folder, exist_ok=True)

# ---------- read vrf ---------- #
# ----- glob and copy input vrf ----- #
sFolder = r"D:\image_database\jn1_mfnr_bestshot\unpacked"
assert os.path.exists(sFolder), f"Data folder does not exist: {sFolder}"
idxVrf, ISO = 53, 6400
vrf_files = glob.glob(os.path.join(sFolder, f"{idxVrf}/*.vrf"))
assert len(vrf_files) > 0, f"VRF file does not exist in folder: {os.path.join(sFolder, str(idxVrf))}"
assert len(vrf_files) == 1, f"Multiple VRF files found in folder: {os.path.join(sFolder, str(idxVrf))}"
sVrfPath = os.path.join(sFolder, vrf_files[0])

sVrfCpyName = f"{idxVrf}_noisy.vrf"
sVrfCpyPath =  os.path.join(sOut_folder, sVrfCpyName)
shutil.copy(sVrfPath, sVrfCpyPath)

# ----- read vrf info ----- #
vrfCur = vrf(sVrfPath)
ISO = vrfCur.m_ISO
print(f"Using ISO: {ISO}")

sVrfOutName = f"{idxVrf}_denoise.vrf"
sVrfOutPath =  os.path.join(sOut_folder, sVrfOutName)

black_level = 64
white_level = 1023
dgain = 1.0

# ----- read vrf ----- #
bayer01_GRBG_noisy = read_vrf(sVrfPath, vrfCur.m_W, vrfCur.m_H, black_level, dgain, white_level)
bayer01_RGGB_noisy = np.fliplr(bayer01_GRBG_noisy)
bayer01_RGGB_noisy = torch.from_numpy(np.ascontiguousarray(bayer01_RGGB_noisy)).cuda(device)

# ---------- Denoise ---------- #
# ----- padd to 32 multiple ----- #
input_rggb_01 = RawUtils.bayer2rggb(bayer01_RGGB_noisy)
input_rggb_01 = input_rggb_01.permute(2, 0, 1).unsqueeze(0)  # [1,4,H,W]

B, C, H, W = input_rggb_01.shape
pad_h = (32 - H % 32) % 32
pad_w = (32 - W % 32) % 32
input_rggb_01 = torch.nn.functional.pad(input_rggb_01, (0, pad_w, 0, pad_h), mode='constant', value = 0)

# ----- get variance map and concate ----- #
kSigma = KSigma(
    K_coeff=Official_Ksigma_params["K_coeff"],
    B_coeff=Official_Ksigma_params["B_coeff"],
    anchor=Official_Ksigma_params["anchor"]
)

k, sigma = kSigma.GetKSigma(iso = ISO)
k = k / 1024
sigma = sigma / 1024 / 1024
variance_map = torch.sqrt(input_rggb_01 * k + sigma).to(torch.float32).to(device)
input_rggb_01_concat = torch.cat([input_rggb_01.to(torch.float32), variance_map], dim=1)

# ----- forward ----- #
pred_rggb_01 = net(input_rggb_01_concat)[0]  # [B,4,H,W]

# ----- depad ----- #
pred_rggb_01 = pred_rggb_01[:, :H, :W]
pred_bayer_01 = RawUtils.rggb2bayer(pred_rggb_01.permute(1, 2, 0)).detach().cpu().numpy()

# ---------- save image ---------- #
# ----- save png ----- #
# bayer01_BGGR_denoise = RawUtils.rggb2bggr(pred_bayer_01)
# bgr01_denoise = RawUtils.bayer01_2_rgb01(bayer01_BGGR_denoise, wb_gain=[1.5156, 1.0, 1.7421], CCM=np.eye(3))
# bgr_denoise = (bgr01_denoise*255.0).astype(np.uint8)
# cv2.imwrite("denoise_rgb.bmp", bgr_denoise)

# ----- cal psnr ----- #
# import skimage
# psnr = skimage.metrics.peak_signal_noise_ratio(bgr_denoise, bgr_noisy)
# print("psnr_bgr = ", psnr)
# print(bayer01_RGGB_denoise.min(), bayer01_RGGB_denoise.max())
# print(bayer01_RGGB_noisy.min(), bayer01_RGGB_noisy.max())
# psnr = skimage.metrics.peak_signal_noise_ratio(bayer01_RGGB_denoise, bayer01_RGGB_noisy) 
# print("psnr_bayer01 = ", psnr)

# bgr_denoise_std = cv2.imread("denoise_rgb_std.bmp")
# errNorm2 = np.linalg.norm(bgr_denoise.astype(np.float32) - bgr_denoise_std.astype(np.float32))
# print("errNorm2 = ", errNorm2)

# ----- save vrf ----- #
out_ratio = 4  #out 12bit
out_black_level = black_level * out_ratio  # 根据实际情况调整
out_white_level = (white_level + 1) * out_ratio - 1
bayer01_GRBG_denoise = np.fliplr(pred_bayer_01)
denoised_image = save_raw_image(bayer01_GRBG_denoise, sVrfOutPath.replace(".vrf", ".raw"), out_white_level, out_black_level)
save_vrf_image(denoised_image, sVrfPath, sVrfOutPath, out_white_level)