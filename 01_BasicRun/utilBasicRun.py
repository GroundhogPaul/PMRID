import cv2
import sys
from pathlib import Path
import os
import numpy as np

# ----- add the parent folder to environment ----- 
ROOT = Path(__file__).resolve().parents[1]
ROOT = Path(__file__).parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

import utilBasic

# ----- read in an image and return normalized RGB with shape (1, CH, H, W) -----
def read_image_BChHW(sImagePath, bNormalize = False, bSwapRB = False, bChHW = False): 
    # TODO: padding, not ready yet
    assert os.path.exists(sImagePath), f"Image path does not exist: {sImagePath}"
   
    img = cv2.imread(sImagePath)
    if img is None:
        raise FileNotFoundError(f"Image not found: {sImagePath}")
    if bSwapRB:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
    if bNormalize:
        img = img.astype('float32') / 255.0
    if bChHW:
        img = img.transpose((2, 0, 1))  # Change to CxHxW
        img = img[None, ...]  # Add batch dimension
        img = np.ascontiguousarray(img)
    return img