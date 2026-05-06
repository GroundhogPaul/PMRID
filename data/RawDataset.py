import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)
import utilData

import cv2
import torch
from torch.utils.data import Dataset

class RawDatasetBasic(Dataset):
    def __init__(self, lstImgPattern, Hbayer, Wbayer, device, CropDataset):
        self.CropDataset = CropDataset(lstImgPattern, Hbayer, Wbayer, device = device)
        self.device = device

    def __len__(self):
        return len(self.CropDataset)

def DumpBgr888(RawDataset, BChHWs, meta_datas, idxDump, imgNameDump, FolderDump):
    assert issubclass(type(RawDataset), RawDatasetBasic)
    assert isinstance(BChHWs, torch.Tensor)
    assert BChHWs.dim() == 3 or BChHWs.dim() == 4, "BChHW_noisy should be a 3D or 4D tensor"
    if BChHWs.dim() == 4:
        assert isinstance(meta_datas, list), "meta_datas should be a list when BChHW_noisy is a 4D tensor"
        assert len(meta_datas) == BChHWs.shape[0], "Length of meta_datas should match batch size of BChHW_noisy"
        BChHW= BChHWs[0]  # (C, H, W) for the specified batch index
        meta_data = meta_datas[0]  # meta_data for the specified batch index
    if BChHWs.dim() == 3:
        assert isinstance(meta_datas, dict), f"meta_datas should be a dict when BChHW_noisy is a 3D tensor, current {type(meta_datas)}"
        meta_data = meta_datas
        BChHW = BChHWs

    if os.path.exists(FolderDump) == False:
        os.makedirs(FolderDump, exist_ok=True)

    bgr888 = RawDataset.ChHW4n_2_bgr888(BChHW, meta_data)
    cv2.imwrite(os.path.join(FolderDump, f"{idxDump:03d}_{imgNameDump}.jpg"), bgr888)