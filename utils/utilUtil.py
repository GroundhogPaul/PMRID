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