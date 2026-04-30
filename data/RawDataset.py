import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

import utilData
import imageio

import glob
import numpy as np
import torch
from RawContainer.utilRaw import RawUtils 
from RawContainer.utilVrf import vrf, CFAPatternEnum, read_vrf, save_vrf_image, save_raw_image 
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
from utils.KSigma import KSigma, Official_Ksigma_params
from utils.NLM import NLM_rggb_withPad
import torchvision.transforms as tvtransforms
import copy
import cv2
import math
import time
from utilBasic import print_gpu_memory_stats

from ImgDataset import CropDatasetVrf, CropDatasetJpg
from AddNoise import add_noise_Concat, add_noise_KSigma

class NumpyHW2TorchChHW:
    """[H,W] ndarray ---> [1,H,W] Tensor"""
    def __call__(self, arr):
        return torch.from_numpy(arr).float().unsqueeze(0)  # [1, H, W]

class RawDatasetTimBrooks(Dataset):
    def __init__(self, lstImgPattern, Hbayer, Wbayer, device, CropDataset):
        self.CropDataset = CropDataset(lstImgPattern, Hbayer, Wbayer, device = device)
        self.device = device

    def __getitem__(self, idx):
        HWCh_rggb_01, mete_data = self.CropDataset[idx]
        ChHW_rggb_01 = HWCh_rggb_01.permute(2, 0, 1)
        noisy, noise_var = add_noise_Concat(ChHW_rggb_01)
        gt = ChHW_rggb_01
        return gt, noisy, noise_var, mete_data
    
    def __len__(self):
        return len(self.CropDataset)

    def ChHW4n_2_bgr888(self, ChHW4n, meta_data, bCPU=True):
        assert isinstance(meta_data, dict)
        HWCh4n = ChHW4n.permute(1, 2, 0)
        bgr888 = self.CropDataset.HWCh4n_2_bgr888(HWCh4n, meta_data, bCPU = bCPU)
        return bgr888

def my_collate(batch):
    imgs = torch.stack([b[0] for b in batch])
    noisys = torch.stack([b[1] for b in batch])
    vars = torch.stack([b[2] for b in batch])
    metas = [b[3] for b in batch]
    return imgs, noisys, vars, metas

def create_dataloader(dataset, batch_size, num_workers=0):
    """Creates a DataLoader for unprocessing training.
    
    Args:
        dir_pattern: A string representing source data directory glob.
        height: Height to crop images.
        width: Width to crop images.
        batch_size: Number of training examples per batch.
        num_workers: Number of workers for parallel data loading.
        
    Returns:
        A PyTorch DataLoader instance.
    """
    prefetch_factor = None
    # prefetch_factor = num_workers if num_workers > 0 else None
    # prefetch_factor = num_workers * 2 if num_workers > 0 else None
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,  # To ensure consistent batch sizes

        num_workers = num_workers,
        prefetch_factor = prefetch_factor,
        persistent_workers = num_workers > 0,

        # pin_memory=True,
        # pin_memory_device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu'),
        
        # timeout=100000,
        collate_fn = my_collate,
    )

if __name__ == "__main__":
    seed = 39
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = 'cuda:2'
    device = torch.device(device if torch.cuda.is_available() else 'cpu')

    # ----- test vrf ----- #
    # Hbayer, Wbayer = 1024, 1024
    # dir_pattern = "D:/image_database/SID/SID/Sony/longVRFmini/*.vrf"
    # dataset = RawDatasetBasic(dir_pattern, Hbayer, Wbayer, device=device, CropDataset=CropDatasetVrf)
    # train_loader = create_dataloader(dataset, 1, num_workers=0)

    # ----- test jpg ----- #
    Hbayer, Wbayer = 512, 512
    dir_pattern = "D:/image_database/mirflickr25k/mirflickr/*.jpg"
    dataset = RawDatasetTimBrooks(dir_pattern, Hbayer, Wbayer, device=device, CropDataset=CropDatasetJpg)
    train_loader = create_dataloader(dataset, 4, num_workers=0)

    for batch_idx, (BChHW_gt, BChHW_noisy, BChHW_var, meta_datas) in enumerate(train_loader):

        bgr888_gt = dataset.ChHW4n_2_bgr888(BChHW_gt[0], meta_datas[0], True)
        cv2.imwrite("RawDatasetTimBrooks_gt.jpg", bgr888_gt)

        bgr888_noisy = dataset.ChHW4n_2_bgr888(BChHW_noisy[0], meta_datas[0], True)
        cv2.imwrite("RawDatasetTimBrooks_noisy.jpg", bgr888_noisy)

        bgr888_var = dataset.ChHW4n_2_bgr888(BChHW_var[0], meta_datas[0], True)
        cv2.imwrite("RawDatasetTimBrooks_var.jpg", bgr888_var)

        break