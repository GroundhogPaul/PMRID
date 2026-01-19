'''to study the purple fringe problem'''

import utilStudy
from utilVrf import vrf, read_vrf, save_vrf_image, save_raw_image
from utilRaw import RawUtils
import os
import numpy as np
import torch

device = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')
sVrfFolder = r"D:\users\xiaoyaopan\PxyAI\PMRID_OFFICIAL\PMRID\02_Study\00_Report20251210\6_TODO2_ColorCast"
Wstart, Hstart = 1372, 1788
Wend, Hend = 2048, 2044

sVrfPath = os.path.join(sVrfFolder, "53_noisy.vrf")
vrfCur = vrf(sVrfPath)
bayer01_GRBG_noisy = read_vrf(sVrfPath, vrfCur.m_W, vrfCur.m_H, black_level=64, dgain=1.0, white_level=1023)
bayer01_GRBG_noisy = bayer01_GRBG_noisy[Hstart:Hend, Wstart:Wend]
bayer01_RGGB_noisy = np.fliplr(bayer01_GRBG_noisy)
bayer01_RGGB_noisy = torch.from_numpy(np.ascontiguousarray(bayer01_RGGB_noisy)).cuda(device)
input_rggb_01_noisy = RawUtils.bayer2rggb(bayer01_RGGB_noisy)

sVrfPath = os.path.join(sVrfFolder, "53_golden_4T.vrf")
vrfCur = vrf(sVrfPath)
bayer01_GRBG_golden4T = read_vrf(sVrfPath, vrfCur.m_W, vrfCur.m_H, black_level=64, dgain=1.0, white_level=1023)
bayer01_GRBG_golden4T = bayer01_GRBG_golden4T[Hstart:Hend, Wstart:Wend]
bayer01_RGGB_golden4T = np.fliplr(bayer01_GRBG_golden4T)
bayer01_RGGB_golden4T = torch.from_numpy(np.ascontiguousarray(bayer01_RGGB_golden4T)).cuda(device)
input_rggb_01_golden4T = RawUtils.bayer2rggb(bayer01_RGGB_golden4T)

sVrfPath = os.path.join(sVrfFolder, "53_denoise_pretrain.vrf")
vrfCur = vrf(sVrfPath)
bayer01_GRBG_pretrain = read_vrf(sVrfPath, vrfCur.m_W, vrfCur.m_H, black_level=64, dgain=1.0, white_level=1023)
bayer01_GRBG_pretrain = bayer01_GRBG_pretrain[Hstart:Hend, Wstart:Wend]
bayer01_RGGB_pretrain = np.fliplr(bayer01_GRBG_pretrain)
bayer01_RGGB_pretrain = torch.from_numpy(np.ascontiguousarray(bayer01_RGGB_pretrain)).cuda(device)
input_rggb_01_pretrain = RawUtils.bayer2rggb(bayer01_RGGB_pretrain)

sVrfPath = os.path.join(sVrfFolder, "53_denoise_505500.vrf")
vrfCur = vrf(sVrfPath)
bayer01_GRBG_PXY = read_vrf(sVrfPath, vrfCur.m_W, vrfCur.m_H, black_level=64, dgain=1.0, white_level=1023)
bayer01_GRBG_PXY = bayer01_GRBG_PXY[Hstart:Hend, Wstart:Wend]
bayer01_RGGB_PXY = np.fliplr(bayer01_GRBG_PXY)
bayer01_RGGB_PXY = torch.from_numpy(np.ascontiguousarray(bayer01_RGGB_PXY)).cuda(device)
input_rggb_01_PXY = RawUtils.bayer2rggb(bayer01_RGGB_PXY)

noisy_res_golden4T = input_rggb_01_noisy - input_rggb_01_golden4T
mean_golden4T = noisy_res_golden4T.mean()
channal_means_golden4T = torch.mean(noisy_res_golden4T, dim=[0,1])
noisy_res_pretrain = input_rggb_01_noisy - input_rggb_01_pretrain
mean_pretrain = noisy_res_pretrain.mean()
channal_means_pretrain = torch.mean(noisy_res_pretrain, dim=[0,1])
noisy_res_PXY = input_rggb_01_noisy - input_rggb_01_PXY
mean_PXY = noisy_res_PXY.mean()
channal_means_PXY = torch.mean(noisy_res_PXY, dim=[0,1])

print(f"Mean of Noisy - Golden4T: {mean_golden4T.item():.6f}")
print(f"Channal Means of Noisy - Golden4T: R: {channal_means_golden4T[0].item():.6f}, G1: {channal_means_golden4T[1].item():.6f}, G2: {channal_means_golden4T[2].item():.6f}, B: {channal_means_golden4T[3].item():.6f}")
print(f"Mean of Noisy - Pretrain: {mean_pretrain.item():.6f}")
print(f"Channal Means of Noisy - Pretrain: R: {channal_means_pretrain[0].item():.6f}, G1: {channal_means_pretrain[1].item():.6f}, G2: {channal_means_pretrain[2].item():.6f}, B: {channal_means_pretrain[3].item():.6f}")
print(f"Mean of Noisy - PXY: {mean_PXY.item():.6f}")
print(f"Channal Means of Noisy - PXY: R: {channal_means_PXY[0].item():.6f}, G1: {channal_means_PXY[1].item():.6f}, G2: {channal_means_PXY[2].item():.6f}, B: {channal_means_PXY[3].item():.6f}")

# ----- calculate channel mean ----- #