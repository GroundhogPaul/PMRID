import utilUtil
import numpy as np
import cv2
import os
import math
import torch
import torch.nn.functional as F
from copy import deepcopy
import time

from utilVrf import vrf, read_vrf, save_vrf_image, save_raw_image, CFAPatternEnum, FlipBayerPattern2Pattern

wSearchWin = 17
wCenterLuma = 5
rPad = wSearchWin // 2
wLumaWin = wSearchWin - 1 # the Luma Result
wPatternWin = 4 # SAD on LumaWin
assert (wLumaWin - wPatternWin) % 2 == 0
arrLumaRoiOffset = (wLumaWin - wPatternWin) // 2
arrCenterLumaOffset = (wSearchWin - 2+1 - wCenterLuma) // 2
pixelOffset = (4 + 2-1) // 2 # 4: 4*4 kernel SAS; 2-1: 2*2 mean to get luma ; forms an efficient 5*5 kernel, and the offset to get center pixel is 2

def NLM_rggb(bayer_rggb_pad):
    assert len(bayer_rggb_pad.shape) == 2
    assert bayer_rggb_pad.shape[0] % 2 == 0
    assert bayer_rggb_pad.shape[1] % 2 == 0

    device = bayer_rggb_pad.device
    dtype = bayer_rggb_pad.dtype

    Hpad, Wpad = bayer_rggb_pad.shape   # 528, 528 
    Hout, Wout = Hpad - rPad - rPad, Wpad - rPad - rPad # 512, 512
    assert Hout % 2 == 0
    assert Wout % 2 == 0
    shape_out = (Hout, Wout)
    bayer_out = torch.zeros(shape_out, dtype = dtype, device = device)

    # ----- Luma ----- #
    arrLuma = bayer_rggb_pad[0:-1, 0:-1] + bayer_rggb_pad[0:-1, 1:] + bayer_rggb_pad[1:, 0:-1] + bayer_rggb_pad[1:, 1:]
    # arrLuma 527, 527

    # ---- Get Diff image ----- #
    arrOfarrLumaDiff = []
    Ndiff_1D = wLumaWin - wPatternWin + 1
    Hpad_Luma, Wpad_Luma = arrLuma.shape # 527, 527
    arrLuma_pm = arrLuma[arrLumaRoiOffset:Hpad_Luma - arrLumaRoiOffset, arrLumaRoiOffset: Wpad_Luma - arrLumaRoiOffset] # 515, 515
    for i in range(Ndiff_1D): # [0,13]
        lstOfarrLumaDiff = []
        if i % 2 == 1:
            arrOfarrLumaDiff.append(lstOfarrLumaDiff)
            continue
        for j in range(Ndiff_1D): # [0,13]
            if j % 2 == 1:
                lstOfarrLumaDiff.append(torch.empty((1,1)))
                continue
            arrLumaDiff = torch.abs(arrLuma[i:Hpad_Luma - 2*arrLumaRoiOffset + i, j:Wpad_Luma - 2*arrLumaRoiOffset + j] - arrLuma_pm)
            H_LumaDiff, W_LumaDiff = arrLumaDiff.shape
            arrLumaFilt = torch.zeros((H_LumaDiff - wPatternWin + 1, W_LumaDiff - wPatternWin + 1), device = device, dtype = dtype)
            for ii in range(wPatternWin):
                for jj in range(wPatternWin):
                    arrLumaFilt += arrLumaDiff[ii:H_LumaDiff - wPatternWin + ii + 1, jj:W_LumaDiff - wPatternWin + jj + 1]
            lstOfarrLumaDiff.append(arrLumaFilt)
        arrOfarrLumaDiff.append(lstOfarrLumaDiff)

    # ----- Get Center Luma ----- #
    arrCenterLuma = torch.zeros((Hout, Wout), device = device, dtype=dtype)
    for i in range(wCenterLuma):
        for j in range(wCenterLuma):
            arrCenterLuma += arrLuma[arrCenterLumaOffset+i:arrCenterLumaOffset+i+Hout, arrCenterLumaOffset+j:arrCenterLumaOffset+j+Wout]
    arrCenterLuma /= (wCenterLuma * wCenterLuma)
    
    # ----- add up diff arr ----- #
    arr_weightSum = torch.zeros(shape_out, device = device, dtype = dtype)
    arr_valueSum = torch.zeros(shape_out, device = device, dtype = dtype) 
    for i in range(0, Ndiff_1D, 2):
        distRow = abs(i - Ndiff_1D//2)
        for j in range(0, Ndiff_1D, 2):
            distCol = abs(j - Ndiff_1D//2)
            weight_spatial = math.exp(-(distRow*distRow + distCol*distCol) / (2 * Ndiff_1D * Ndiff_1D / 3))
            # --- weight ---
            arr_diff = arrOfarrLumaDiff[i][j]
            arr_diff /= arrCenterLuma  # weight_Luma
            arr_weight = 1 / ( arr_diff +  0.1)  # TODO most naive method, not ready yet, the curve in in ISP01 and in C code
            arr_weight *= weight_spatial
            arr_weightSum += arr_weight

            # --- pixel value weighted --- #
            arr_value = bayer_rggb_pad[pixelOffset+i:pixelOffset+i+Hout,pixelOffset+j:pixelOffset+j+Wout]
            arr_valueSum += arr_weight * arr_value
    
    # ----- average ----- #
    bayer_out = arr_valueSum / arr_weightSum
    bayer_out = torch.round(bayer_out).to(torch.int16)

    return bayer_out

def NLM_rggb_withPad(bayer_rggb_noisy):
    # ---------- pad the vrf ---------- #
    bayer_pad = F.pad(bayer_rggb_noisy, (rPad, rPad, rPad, rPad), mode = 'reflect')
    bayer_pad = bayer_pad.to(torch.float32)
    bayer_pad = bayer_pad[0]
    bayer_out = NLM_rggb(bayer_pad)
    return bayer_out

if __name__ == "__main__":

    # ---------- get vrf ---------- #
    sVrfPath = "D:/image_database/jn1_mfnr_bestshot/unpacked/33/5_unpacked.vrf"
    # sVrfPath = "./testIn.vrf"
    # sVrfPath = "./4K.vrf"
    # sVrfPath = "./1_AI_Denoise.vrf"
    assert os.path.exists(sVrfPath), f"sVrfPath does not exist: {sVrfPath}"

    vrfCur = vrf(sVrfPath)
    bayer_noisy = vrfCur.get_raw_image()

    # ---------- flip and to torch --------- #
    bayer_RGGB_noisy = FlipBayerPattern2Pattern(bayer_noisy, vrfCur.m_CFAPatternNum, CFAPatternEnum.RGGB)
    bayer_RGGB_noisy = torch.from_numpy(np.ascontiguousarray(bayer_RGGB_noisy)).unsqueeze(0)

    device = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')
    bayer_RGGB_noisy = bayer_RGGB_noisy.to(device)

    for iFrame in range(1):

        start_time = time.time()
        bayer_out = NLM_rggb_withPad(bayer_RGGB_noisy)
        end_time = time.time()

        print(f'NLM spent time: {(end_time - start_time):.2f}s')

    # ---------- to numpy and flip --------- #
    bayer_out = bayer_out.cpu().numpy()
    bayer_out = FlipBayerPattern2Pattern(bayer_out, CFAPatternEnum.RGGB, vrfCur.m_CFAPatternNum)

    # ---------- save the vrf ---------- #
    vrfOut = deepcopy(vrfCur)
    vrfOut.m_raw = bayer_out

    vrfOut.m_H = bayer_out.shape[0]
    vrfOut.m_W = bayer_out.shape[1]
    vrfOut.save_vrf("testOut.vrf")

    print("done")