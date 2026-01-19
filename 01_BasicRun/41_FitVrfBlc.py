import utilBasicRun

import numpy as np
import skimage
import os
import cv2
from utilRaw import RawUtils
from run_benchmark import KSigma, Official_Ksigma_params
from utilVrf import vrf, read_vrf, save_vrf_image, save_raw_image 
from models.net_torch import NetworkSingleNoise as Network
import torch
import shutil
import glob
from PlotAgainShotRead import interpolate_gain_var, GetJin1ShotAndReadVar


sFolder = r"D:\users\xiaoyaopan\PxyAI\DataSet\Jn1\s5kjn1_noise_calibration_raw"
sFileName = r"optical_black/64x_unpack.vrf"
# sFileName = r"noise_ccm/ccm_64x_1.vrf"

# sFolder = r"D:\users\xiaoyaopan\PxyAI\DataSet\Jn1\s5kjn1_calibration_raw"
# sFileName = r"blc_unpack/gain_64_unpack.vrf"

assert os.path.exists(sFolder), f"Data folder does not exist: {sFolder}"
sVrfPath = os.path.join(sFolder, sFileName)
assert os.path.exists(sVrfPath), f"Data file does not exist: {sVrfPath}"

vrfCur = vrf(sVrfPath)
ISO = vrfCur.m_ISO
SensorGain = vrfCur.m_nSensorGain
print(f"Using ISO: {ISO}, SensorGain: {SensorGain}")


black_level = vrfCur.m_BlackLevel
white_level = vrfCur.m_WhiteLevel 
blc01 = float(black_level) / white_level
dgain = 1.0

# ----- read vrf ----- #
noisy_bayerGRBG = read_vrf(sVrfPath, vrfCur.m_W, vrfCur.m_H, 0, dgain, white_level)
noisy_bayerRGGB = np.fliplr(noisy_bayerGRBG)

# 计算均值和方差
mean_value = np.mean(noisy_bayerRGGB)
variance_value = np.var(noisy_bayerRGGB)

print(f"图像均值: {mean_value}")
print(f"图像方差: {variance_value}")

sig = np.sqrt(variance_value)

outlier_3sig_pos = mean_value + 3*sig
outlier_3sig_neg = mean_value - 3*sig

arr_outlier_3sig_pos = noisy_bayerRGGB > outlier_3sig_pos
arr_outlier_3sig_neg = noisy_bayerRGGB < outlier_3sig_neg
N_outlier_3sig_pos = np.count_nonzero(arr_outlier_3sig_pos)
N_outlier_3sig_neg = np.count_nonzero(arr_outlier_3sig_neg)

print(f"非零元素个数: {N_outlier_3sig_pos}, {N_outlier_3sig_neg}")


mean_value = np.mean(noisy_bayerRGGB[:256, :256])
variance_value = np.var(noisy_bayerRGGB[:256, :256])

print(f"图像均值: {mean_value}")
print(f"图像方差: {variance_value}")

mean_value = np.mean(noisy_bayerRGGB[1024:, 1024:])
variance_value = np.var(noisy_bayerRGGB[1024:, 1024:])

print(f"图像均值: {mean_value}")
print(f"图像方差: {variance_value}")

print("done")
