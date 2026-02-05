import utilUtil
import numpy as np
import cv2
import os
import torch
import torch.nn.functional as F
from copy import deepcopy
import time

from utilVrf import vrf, read_vrf, save_vrf_image, save_raw_image, CFAPatternEnum, FlipBayerPattern2Pattern

wSearchWin = 17
rPad = wSearchWin // 2
wLumaWin = 16 # the Luma Result
wPatternWin = 4 # SAD on LumaWin

def NLM_rggb(bayer_rggb_pad):
    assert len(bayer_rggb_pad.shape) == 2
    assert bayer_rggb_pad.shape[0] % 2 == 0
    assert bayer_rggb_pad.shape[1] % 2 == 0

    Hpad, Wpad = bayer_rggb_pad.shape   # 528, 528 
    Hout, Wout = Hpad - rPad - rPad, Wpad - rPad - rPad # 512, 512
    assert Hout % 2 == 0
    assert Wout % 2 == 0
    shape_out = (Hout, Wout)
    bayer_out = torch.zeros(shape_out, dtype = bayer_rggb_pad.dtype, device = bayer_rggb_pad.device)

    # ----- Luma ----- #
    arrLuma = bayer_rggb_pad[0:-1, 0:-1] + bayer_rggb_pad[0:-1, 1:] + bayer_rggb_pad[1:, 0:-1] + bayer_rggb_pad[1:, 1:]
    # arrLuma 527, 527

    # ---- Get Diff image ----- #
    arrOfarrLumaDiff = []
    Ndiff_1D = wLumaWin - wPatternWin + 1
    Hpad_Luma, Wpad_Luma = arrLuma.shape # 527, 527
    arrLuma_pm = arrLuma[6:Hpad_Luma - 6, 6: Wpad_Luma - 6] # 515, 515
    for i in range(Ndiff_1D): # [0,13]
        lstOfarrLumaDiff = []
        if i % 2 == 1:
            arrOfarrLumaDiff.append(lstOfarrLumaDiff)
            continue
        for j in range(Ndiff_1D): # [0,13]
            if j % 2 == 1:
                lstOfarrLumaDiff.append(torch.empty((1,1)))
                continue
            arrLumaDiff = torch.abs(arrLuma[i:Hpad_Luma - 6 - 6 + i, j:Wpad_Luma - 6 - 6 + j] - arrLuma_pm)
            H_LumaDiff, W_LumaDiff = arrLumaDiff.shape
            arrLumaFilt = torch.zeros((H_LumaDiff - 4 + 1, W_LumaDiff - 4 + 1))
            for ii in range(4):
                for jj in range(4):
                    arrLumaFilt += arrLumaDiff[ii:H_LumaDiff - 4 + ii + 1, jj:W_LumaDiff - 4 + jj + 1]
            lstOfarrLumaDiff.append(arrLumaFilt)
        arrOfarrLumaDiff.append(lstOfarrLumaDiff)
    
    # ----- Diff Weight ----- # # TODO currently most naive way
    arrOfarrLumaWeight = deepcopy(arrOfarrLumaDiff)
    for i in range(Ndiff_1D):
        for j in range(Ndiff_1D):
            if i % 2 == 0 and j % 2 == 0:
                arrOfarrLumaWeight[i][j] = 1 / (arrOfarrLumaDiff[i][j] + 10)
    
    # ----- add up diff arr ----- #
    arr_weightSum = torch.zeros(shape_out, dtype = float)
    arr_valueSum = torch.zeros(shape_out, dtype = float) 
    for i in range(0, Ndiff_1D, 2):
        for j in range(0, Ndiff_1D, 2):
            arr_value = bayer_rggb_pad[2+i:2+i+Hout,2+j:2+j+Wout]
            arr_weight = arrOfarrLumaWeight[i][j]
            arr_weightSum += arr_weight
            arr_valueSum += arr_weight * arr_value
    
    # ----- average ----- #
    bayer_out = arr_valueSum / arr_weightSum
    bayer_out = torch.round(bayer_out).to(torch.int16)

    return bayer_out

if __name__ == "__main__":

    # ---------- get vrf ---------- #
    # sVrfPath = "D:/image_database/jn1_mfnr_bestshot/unpacked/33/5_unpacked.vrf"
    # sVrfPath = "./testIn.vrf"
    sVrfPath = "./1_AI_Denoise.vrf"
    assert os.path.exists(sVrfPath), f"sVrfPath does not exist: {sVrfPath}"

    vrfCur = vrf(sVrfPath)
    bayer_noisy = vrfCur.get_raw_image()

    # ---------- flip and to torch --------- #
    bayer_RGGB_noisy = FlipBayerPattern2Pattern(bayer_noisy, vrfCur.m_CFAPatternNum, CFAPatternEnum.RGGB)
    bayer_RGGB_noisy = torch.from_numpy(np.ascontiguousarray(bayer_RGGB_noisy)).unsqueeze(0)

    # ---------- pad the vrf ---------- #
    bayer_pad = F.pad(bayer_RGGB_noisy , (rPad, rPad, rPad, rPad), mode = 'reflect')
    bayer_pad = bayer_pad[0]
    bayer_pad_ROI = bayer_pad[rPad:, rPad:]

    # ---------- TODO: NLM ---------- #
    start_time = time.time()
    bayer_out = NLM_rggb(bayer_pad)
    end_time = time.time()
    print(f'NLM spent time: {(end_time - start_time):.2f}s')

    # ---------- to numpy and flip --------- #
    bayer_out = bayer_out.numpy()
    bayer_out = FlipBayerPattern2Pattern(bayer_out, CFAPatternEnum.RGGB, vrfCur.m_CFAPatternNum)

    # ---------- save the vrf ---------- #
    vrfOut = deepcopy(vrfCur)
    vrfOut.m_raw = bayer_out

    vrfOut.m_H = bayer_out.shape[0]
    vrfOut.m_W = bayer_out.shape[1]
    vrfOut.save_vrf("testOut.vrf")

    print("done")