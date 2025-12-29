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
from PlotAgainShotRead import interpolate_gain_var

# ---------- read model ---------- #
# ----- assert ckpt paths ----- #
# model_path =  "./models/TIM_BROOKS_test/top_models/top_model_psnr_50.13_epoch_3910.pth"
model_path =  "./models/TIM_BROOKS_test_AsMuchBlc/top_models/top_model_psnr_50.30_epoch_7180.pth"
# model_path =  "./models/TIM_BROOKS_test_AsMuchBlc_AWB/top_models/top_model_psnr_49.74_epoch_530.pth"
# model_path =  "./models/TIM_BROOKS_test_AsMuchBlc_NegativeGT/top_models/lateset_model_psnr_0.00_epoch_7400.pth"
assert os.path.exists(model_path), f"Model file does not exist: {model_path}"

# ----- get model name -----
path_parts = model_path.split('/')
models_index = path_parts.index('models')
sImgSuffix = path_parts[models_index + 1]
print("sImgSuffix = ", sImgSuffix)
assert os.path.exists(model_path), f"Model file does not exist: {model_path}"

# ----- load ckpt ----- #
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
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
idxVrf = 53
# idxVrf = 33
# idxVrf = 1
vrf_files = glob.glob(os.path.join(sFolder, f"{idxVrf}/*.vrf"))
assert len(vrf_files) > 0, f"VRF file does not exist in folder: {os.path.join(sFolder, str(idxVrf))}"
assert len(vrf_files) == 1, f"Multiple VRF files found in folder: {os.path.join(sFolder, str(idxVrf))}"
sVrfPath = os.path.join(sFolder, vrf_files[0])

sVrfCpyName = f"{idxVrf:02d}_noisy.vrf"
sVrfCpyPath =  os.path.join(sOut_folder, sVrfCpyName)
shutil.copy(sVrfPath, sVrfCpyPath)

# ----- read vrf info ----- #
vrfCur = vrf(sVrfPath)
ISO = vrfCur.m_ISO
SensorGain = vrfCur.m_nSensorGain
print(f"Using ISO: {ISO}, SensorGain: {SensorGain}")

sVrfOutName = f"{idxVrf:02d}_{sImgSuffix}_denoise.vrf"
sVrfOutPath =  os.path.join(sOut_folder, sVrfOutName)

black_level = 64
white_level = 1023
blc01 = float(black_level) / white_level
dgain = 1.0

# ----- read vrf ----- #
# bayer01_GRBG_noisy = read_vrf(sVrfPath, vrfCur.m_W, vrfCur.m_H, black_level, dgain, white_level)
noisy_bayerGRBG = read_vrf(sVrfPath, vrfCur.m_W, vrfCur.m_H, 0, dgain, white_level)
noisy_bayerRGGB = np.fliplr(noisy_bayerGRBG)
noisy_bayerRGGB = torch.from_numpy(np.ascontiguousarray(noisy_bayerRGGB)).cuda(device)
noisy_bayerRGGB = noisy_bayerRGGB - blc01

Hmin, Hmax = 1010, 1015
Wmin, Wmax = 2628, 2634
# print(bayer_RGGB_noisy[Hmin:Hmax+1, Wmin:Wmax+1])
print(noisy_bayerRGGB.min(), noisy_bayerRGGB.max())

# ---------- Denoise ---------- #
# ----- padd to 32 multiple ----- #
noisy_rggb = RawUtils.bayer2rggb(noisy_bayerRGGB)

noisy_rggb = noisy_rggb.permute(2, 0, 1).unsqueeze(0)  # [1,4,H,W]

B, C, H, W = noisy_rggb.shape
pad_h = (32 - H % 32) % 32
pad_w = (32 - W % 32) % 32
noisy_rggb = torch.nn.functional.pad(noisy_rggb, (0, pad_w, 0, pad_h), mode='constant', value = 0)

# ----- get variance map and concate ----- #
sFileGain = "D:/image_database/jn1_mfnr_bestshot/gain.txt"
sFileVar = "D:/image_database/jn1_mfnr_bestshot/var.txt"

SensorGain = np.clip(SensorGain / 16 * 0.38, 1, 64)
sigShot, sigRead = interpolate_gain_var(file_gain = sFileGain, file_var = sFileVar, TGain = SensorGain)
print("sigShot = ", sigShot, ", sigRead = ", sigRead)
var_rggb = torch.sqrt(torch.clamp(noisy_rggb, 0, 1) * sigShot + sigRead).to(torch.float32).to(device)
concat_rggb = torch.cat([noisy_rggb.to(torch.float32), var_rggb], dim=1)

# ----- forward ----- #
pred_rggb = net(concat_rggb)[0]  # [B,4,H,W]

# ----- depad ----- #
pred_rggb = pred_rggb[:, :H, :W]
pred_bayerRGGB = RawUtils.rggb2bayer(pred_rggb.permute(1, 2, 0)).detach().cpu().numpy()

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
pred_bayerGRBG = np.fliplr(pred_bayerRGGB)
# bayer01_GRBG_denoise = np.clip(bayer01_GRBG_denoise, 0, 1)
denoised_image = save_raw_image(pred_bayerGRBG, sVrfOutPath.replace(".vrf", ".raw"), out_white_level, out_black_level)
save_vrf_image(denoised_image, sVrfPath, sVrfOutPath, out_white_level)