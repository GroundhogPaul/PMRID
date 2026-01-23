import utilBasicRun
import numpy as np
from glob import glob
import matplotlib.pyplot as plt

import os
from utilVrf import vrf, read_vrf, save_vrf_image, save_raw_image 

def get_vrf_hist(sVrfPath):
    assert os.path.exists(sVrfPath), f"sVrfPath does not exist: {sVrfPath}"

    # ----- read vrf info ----- #
    vrfCur = vrf(sVrfPath)
    ISO = vrfCur.m_ISO
    SensorGain = vrfCur.m_nSensorGain
    bayerRaw = vrfCur.get_raw_image()

    bayerRaw = np.clip(bayerRaw.astype(np.float32) - vrfCur.m_BlackLevel, 0, vrfCur.m_WhiteLevel)
    bayerRaw = bayerRaw / vrfCur.m_WhiteLevel * 255.0
    bayerRaw = bayerRaw.astype(np.uint8)
    hist, bins = np.histogram(bayerRaw, bins=256, range=(0, 255))
    hist = hist.astype(np.float32)

    return hist, bins

if __name__ == "__main__":
    # ---------- case1: single vrf ---------- #
    # sFolder = r"D:/image_database/SID/SID/Sony/longVRF"
    # assert os.path.exists(sFolder), f"Data folder does not exist: {sFolder}"

    # sVrfPath = os.path.join(sFolder, "00012_00_10s.vrf")

    # hist, bins = get_vrf_hist(sVrfPath)
    # hist = hist / np.sum(hist)

    # ---------- case2: batch vrf ---------- #
    lstFolder = ["D:/image_database/SID/SID/Sony/longVRF"]
    lstVrfPath = []
    for pathFolder in lstFolder:
        files = glob(os.path.join(pathFolder, "*.vrf"))
        if not files:
            AssertionError(f"Folder '{pathFolder}' 没有找到文件")
        lstVrfPath.extend(files)
    
    for idx, sVrfPath in enumerate(lstVrfPath):
        print(f"Processing {idx+1}/{len(lstVrfPath)}: {sVrfPath}")
        hist_cur, bins = get_vrf_hist(sVrfPath)
        if idx == 0:
            hist = hist_cur
        else:
            hist += hist_cur
    hist = hist / np.sum(hist)

    # ---------- plot hist ---------- #
    plt.figure(figsize=(10,5))
    plt.plot(bins[:-1], hist, color='blue')
    plt.title("Histogram of Bayer Raw Image")
    plt.xlabel("Pixel Value")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.show()
    print("Done")