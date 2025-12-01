import utilBasicRun
from models.net_torch import *
import matplotlib.pyplot as plt

import numpy as np
import os
import cv2
from utils import RawUtils
from utilDng import *
import pickle

# ---------- collect one ISO dngs ---------- #
sFolder = r"D:\users\xiaoyaopan\PxyAI\DataSet\PMRID\reno10x_noise\gray_scale_chart\RAW_2020_02_20_13_06_43_108"
sFolder = r"D:\users\xiaoyaopan\PxyAI\DataSet\PMRID\reno10x_noise\gray_scale_chart\RAW_2020_02_20_13_10_30_677"
assert os.path.exists(sFolder), f"Data folder does not exist: {sFolder}"
bLoadPickle = True
# bLoadPickle = False
pickle_path = "dng_single_iso_mean_std.pkl"
pickle_path = os.path.join(sFolder, pickle_path)

listDng = []
for fileIth in os.listdir(sFolder):
    if fileIth.lower().endswith('.dng'):
        listDng.append(os.path.join(sFolder, fileIth))

# ---------- read dngs ---------- #
if not bLoadPickle:
    lstBayer = []
    for sDngIth in listDng:
        DngReaderObj = DngReader(sDngIth)
        bayer = DngReaderObj.raw.raw_image.astype(np.float32)
        RGGB = RawUtils.bayer2rggb(bayer)
        # bayer = RGGB[:, :, 0]  # only use R channel 
        lstBayer.append(bayer)

    # ---------- stack ---------- #
    bayer_noisy = np.stack(lstBayer, axis=0)

    # ---------- compute mean and std ---------- #
    E_x = np.mean(bayer_noisy, axis=0) # 得到的矩阵形状为 (3000, 4000)
    # plt.plot(E_x[0])
    # plt.show()
    Std_x = np.std(bayer_noisy, axis=0)   # 得到的矩阵形状也为 (3000, 4000)
    print("平均值矩阵形状:", E_x.shape) # 应为 (3000, 4000)
    print("标准差矩阵形状:", Std_x.shape)   # 应为 (3000, 4000)
    Var_x = Std_x**2

    # ---------- save results ---------- #
    with open(pickle_path, 'wb') as f:
        pickle.dump({'E_x': E_x, 'Std_x': Std_x, 'Var_x': Var_x}, f)

# ---------- read back and verify ---------- #
if bLoadPickle:
    with open(pickle_path, 'rb') as f:
        data = pickle.load(f)
    E_x = data['E_x']
    Var_x = data['Var_x'] 

print("平均值矩阵形状:", E_x.shape) # 应为 (3000, 4000)
print("标准差矩阵形状:", Var_x.shape)   # 应为 (3000, 4000)

print(E_x[0])
print(Var_x[0]) 

# ---------- fit k and sigma ---------- #
from scipy.optimize import curve_fit
def model_func(x, k, sigTo2):
    return k * x + sigTo2

E_x_flat = E_x.flatten()
Var_x_flat = Var_x.flatten()

plt.scatter(E_x_flat[:1000], Var_x_flat[:1000], s=1, alpha=0.5)
plt.show()

popt, pcov = curve_fit(model_func, E_x_flat, Var_x_flat)
k_fit, sigTo2_fit = popt
print(f"拟合结果:")
# print(f"k = {k_fit:.6f} ± {k_err:.6f}")
# print(f"sigTo2 = {sigTo2_fit:.6f} ± {sigTo2_err:.6f}")
print(f"拟合方程: Var_x = {k_fit:.6f} * E_x + {sigTo2_fit:.6f}")



