'''UT test of : load the official PMRID test set'''
import utilBasicRun
from models.net_torch import *
import json

import numpy as np
import os
import cv2
from utils import RawUtils

sFolder = r"D:\users\xiaoyaopan\PxyAI\DataSet\PMRID\PMRID"
assert os.path.exists(sFolder), f"Data folder does not exist: {sFolder}"
sRawKey = "Scene1/dark/RAW_2020_02_27_17_49_05_875/input.raw"
# sRawKey = "Scene2/dark/RAW_2020_02_27_19_35_27_325/input.raw"
# sRawKey = "Scene3/dark/RAW_2020_02_27_19_57_17_976/input.raw"
# sRawKey = "Scene4/dark/RAW_2020_02_27_17_49_05_875/input.raw"
sRawPath = os.path.join(sFolder, sRawKey)
assert os.path.exists(sRawPath), f"Raw file does not exist: {sRawPath}"

sJson = r"D:\users\xiaoyaopan\PxyAI\DataSet\PMRID\PMRID\benchmark.json"
assert os.path.exists(sJson), f"JSON file does not exist: {sJson}"

# with open(sJson, 'r') as f:
#     json_data = json.load(f)

# for imgInfoIth in json_data:
#     if imgInfoIth['input'] == sRawKey:
#         print("Found matching entry in JSON:")
#         imgInfo = imgInfoIth
#         break   

class RawImageLoader:
    def __init__(self):
        pass
    
    def load_from_PMRID_json(self, sJson: str, sRawKey: str):
        # ---------- find info in PMRID json according to sRawKey ----------
        assert os.path.exists(sJson), f"JSON file does not exist: {sJson}"
        with open(sJson, 'r') as f:
            json_data = json.load(f)
        
        imgInfo = None
        for imgInfoIth in json_data:
            if imgInfoIth['input'] == sRawKey:
                imgInfo = imgInfoIth
                break
        assert imgInfo is not None, f"Raw key {sRawKey} not found in JSON."

        # ---------- load parameters ----------
        self.sRawPath = os.path.join(os.path.dirname(sJson), imgInfo['input'])
        assert os.path.exists(self.sRawPath), f"Raw file does not exist: {self.sRawPath}"
        self.sGtPath = os.path.join(os.path.dirname(sJson), imgInfo['gt'])
        assert os.path.exists(self.sGtPath), f"GT file does not exist: {self.sGtPath}"

        meta = imgInfo.get('meta', {})
        self.name = meta['name']
        self.scene_id = meta['scene_id']
        self.light = meta['light']
        self.ISO = meta['ISO']
        self.exp_time = meta['exp_time']
        self.bayer_pattern = meta['bayer_pattern']
        self.H = meta['shape'][0]
        self.W = meta['shape'][1]
        self.wb_gain = meta['wb_gain']
        self.CCM = meta['CCM']
        self.ROIs = meta['ROIs']

        with open(self.sGtPath, 'rb') as f:
            raw_data = f.read()
        self.bayer_BGGR_gt = np.frombuffer(raw_data, dtype=np.uint16)
        self.bayer_BGGR_gt = self.bayer_BGGR_gt.reshape(self.H, self.W)
        assert cv2.imwrite("bayer_BGGR_gt.png", (self.bayer_BGGR_gt / 65535.0 * 255.0).astype(np.uint8))
        
        # ---------- load raw image ----------
        # expected_size = width * height * bpp * num_channels
        with open(self.sRawPath, 'rb') as f:
            raw_data = f.read()
        
        # assert len(raw_data) == expected_size, f"Unexpected raw file size: expected {expected_size}, got {len(raw_data)}"
        
        self.bayer_BGGR = np.frombuffer(raw_data, dtype=np.uint16)
        self.bayer_BGGR = self.bayer_BGGR.reshape(self.H, self.W)
        assert cv2.imwrite("bayer_BGGR.png", (self.bayer_BGGR / 65535.0 * 255.0).astype(np.uint8))
        
        self.BGGR = RawUtils.bayer2rggb(self.bayer_BGGR)
        self.RGGB = RawUtils.bggr2rggb(self.BGGR)
        self.bayer_RGGB = RawUtils.rggb2bayer(self.RGGB)
        # assert cv2.imwrite("bayer_RGGB.png", (self.bayer_RGGB / 65535.0 * 255.0).astype(np.uint8))

        self.RGB = RawUtils.bayer2rgb(self.bayer_RGGB / 65535.0, wb_gain=self.wb_gain, CCM=self.CCM)
        cv2.imwrite("debug_RGB_U8.png", (self.RGB*255.0).astype(np.uint8))
        # assert cv2.imwrite("debug_RGB_U8.png", (self.RGB).astype(np.uint8))
        return


RawImageLoaderObj = RawImageLoader()
RawImageLoaderObj.load_from_PMRID_json(sJson, sRawKey)