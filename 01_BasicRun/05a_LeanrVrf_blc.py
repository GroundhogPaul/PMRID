import utilBasicRun

import os
from utilVrf import vrf, read_vrf, save_vrf_image, save_raw_image 
import numpy as np
import cv2

# ---------- get vrf ---------- #
sFolder = r"./02_Study/06_StudyVrf"
assert os.path.exists(sFolder), f"Data folder does not exist: {sFolder}"

sVrfPath = os.path.join(sFolder, "53_AI_Denoise_Golden4T.vrf")
print(sVrfPath)
assert os.path.exists(sVrfPath), f"sVrfPath does not exist: {sVrfPath}"

# ----- read vrf info ----- #
vrfCur = vrf(sVrfPath)
ISO = vrfCur.m_ISO
SensorGain = vrfCur.m_nSensorGain
bayerRaw = vrfCur.read_raw_raw()
print(f"Using ISO: {ISO}, SensorGain: {SensorGain}")

black_level = vrfCur.m_BlackLevel
underBlc = bayerRaw < black_level
underBlc = underBlc.astype(np.uint8) * 255

sPathJpgOut = os.path.join(sFolder, "underBlc.jpg")
cv2.imwrite(sPathJpgOut, underBlc)
print("jpg saved to : ", sPathJpgOut)