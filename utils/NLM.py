import utilUtil
import numpy as np
import cv2
import os
import torch
import torch.nn.functional as F
from copy import deepcopy

from utilVrf import vrf, read_vrf, save_vrf_image, save_raw_image, CFAPatternEnum, FlipBayerPattern2Pattern

wSearchWin = 17
rPad = wSearchWin // 2
wLumaWin = 16 # the Luma Result
wPatternWin = 4 # SAD on LumaWin

def NLMrggb(bayer_rggb):
    pass


if __name__ == "__main__":

    # ---------- get vrf ---------- #
    sVrfPath = "D:/image_database/jn1_mfnr_bestshot/unpacked/33/5_unpacked.vrf"
    # sVrfPath = "./1_AI_Denoise.vrf"
    assert os.path.exists(sVrfPath), f"sVrfPath does not exist: {sVrfPath}"

    vrfCur = vrf(sVrfPath)
    bayer_noisy = vrfCur.get_raw_image()

    # ---------- flip and to torch --------- #
    bayer_RGGB_noisy = FlipBayerPattern2Pattern(bayer_noisy, vrfCur.m_CFAPatternNum, CFAPatternEnum.RGGB)
    bayer_RGGB_noisy = torch.from_numpy(np.ascontiguousarray(bayer_RGGB_noisy)).unsqueeze(0)

    # ---------- pad the vrf ---------- #
    bayer_pad = F.pad(bayer_RGGB_noisy , (rPad, rPad, rPad, rPad), mode = 'reflect')

    # ---------- TODO: NLM ---------- #


    # ---------- to numpy and flip --------- #
    bayer_out = bayer_pad.squeeze(0).numpy()
    bayer_out = FlipBayerPattern2Pattern(bayer_out, CFAPatternEnum.RGGB, vrfCur.m_CFAPatternNum)

    # ---------- save the vrf ---------- #
    vrfOut = deepcopy(vrfCur)
    vrfOut.m_raw = bayer_out

    vrfOut.m_W = bayer_pad.shape[2]
    vrfOut.m_H = bayer_pad.shape[1]
    vrfOut.save_vrf("testOut.vrf")

    print("done")