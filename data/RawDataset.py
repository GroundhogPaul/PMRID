import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)
import utilData

from torch.utils.data import Dataset

class RawDatasetBasic(Dataset):
    def __init__(self, lstImgPattern, Hbayer, Wbayer, device, CropDataset):
        self.CropDataset = CropDataset(lstImgPattern, Hbayer, Wbayer, device = device)
        self.device = device

    def __len__(self):
        return len(self.CropDataset)