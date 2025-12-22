import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)  # 插入开头，优先搜索

import utilData
import utilBasic
import imageio
import glob
import numpy as np
import torch
from utilRaw import RawUtils
from utilArw import ArwReader
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import torchvision.transforms as tvtransforms
import copy
import cv2
import math
from utilBasic import print_gpu_memory_stats

from RawDataset import RawArrayToTensor, PMRIDRawDataset, create_dataloader 

class TimBrooksRawDataset(PMRIDRawDataset):
    def __init__(self, dir_pattern, height=1024, width=1024, bPreLoadAll=False, device='cpu'):
        super().__init__(dir_pattern, height, width, bPreLoadAll=bPreLoadAll, device=device)

    def add_noise(self, input_bayer_01, noise_type='Gaussian'):
        log_min_shot_noise = torch.log(torch.tensor(0.0001))
        log_max_shot_noise = torch.log(torch.tensor(0.012))
        log_shot_noise =  log_min_shot_noise + (log_max_shot_noise - log_min_shot_noise) * torch.rand(1)
        shot_noise = torch.exp(log_shot_noise).item()

        line = lambda x: 2.18 * x + 1.2
        log_read_noise = line(log_shot_noise) + torch.normal(mean = 0.0, std = 0.26, size=())
        read_noise = torch.exp(log_read_noise).item()

        variance = input_bayer_01 * shot_noise + read_noise
        noise = torch.randn_like(input_bayer_01) * torch.sqrt(variance)
        
        noisy_bayer01 = input_bayer_01 + noise
        return torch.clamp(noisy_bayer01.to(torch.float32), 0, 1), copy.deepcopy(shot_noise), copy.deepcopy(read_noise) 
    
    def __getitem__(self, index):
        # read raw
        if self.bPreLoadAll:
            ArwReaderCur = self.preloaded_raw[index]
            input_bayer = self.preloaded_bayer_device[index] # GPU buffer
        else:
            ArwReaderCur = ArwReader(self.filenames[index])
            # ArwReaderCur = ArwReader("D:/image_database/SID/SID/Sony/long/00002_00_10s.ARW") # test code
            input_bayer = ArwReaderCur.raw_image.astype(np.float32)
            input_bayer = np.ascontiguousarray(input_bayer)
            input_bayer = RawArrayToTensor()(input_bayer)
            input_bayer = input_bayer.to(self.device)
        bayer_pattern = ArwReaderCur.raw_pattern
        white_level = ArwReaderCur.white_level
        black_level = ArwReaderCur.black_level_per_channel[0]
        B, H, W = input_bayer.shape

        # data transform          
        input_bayer = self.random_crop_and_flip(input_bayer, bayer_pattern, H_crop=self.height, W_crop=self.width, p_flip_ud=0.5, p_flip_lr=0.5)
        # input_bayer_01 = input_bayer / white_level  # no black_level at all, copied from benchmark.py.BenchmarkLoader
        # input_bayer_01 = (input_bayer - black_level) / (white_level - black_level)
        input_bayer_01 = (input_bayer - black_level) / white_level

        # brightness and contrast augmentation
        input_bayer_01 = tvtransforms.ColorJitter(brightness=(0.2, 1.2), contrast=(0.5, 1.5))(input_bayer_01)
        input_bayer_01 = torch.clamp(input_bayer_01, 0.0, 1.0)
        input_rggb_01 = RawUtils.bayer_to_rggb(input_bayer_01, "RGGB")  # to [H/2, W/2, 4] RGGB

        # add random noise
        input_rggb_01_noisy, shot_noise, read_noise = self.add_noise(input_rggb_01)
        # print("\n train set: shot_noise:", shot_noise.item(), " read_noise:", read_noise.item())
        input_rggb_variance = torch.sqrt(input_rggb_01_noisy * shot_noise + read_noise)

        # save meta data
        wb_gain = ArwReaderCur.GetWBgain01("RGGB")
        ccm3x3 = ArwReaderCur.GetCCM()
        meta_data = {
            'black_level': black_level,
            'wb_gain': wb_gain,
            'ccm': ccm3x3
        }

        input_rggb_01 = input_rggb_01.permute(2,0,1).to(self.device)
        input_rggb_01_noisy = input_rggb_01_noisy.permute(2,0,1).to(self.device)
        input_rggb_variance = input_rggb_variance.permute(2,0,1).to(self.device)
        return input_rggb_01, input_rggb_01_noisy, input_rggb_variance, meta_data

    @classmethod
    def ConvertDatasetImgToBGR888(cls, inputs_rggb, meta_data, idx = 0):

        input_rggb = inputs_rggb[idx]
        input_rggb = input_rggb.permute(1, 2, 0).to('cpu')

        if hasattr(input_rggb, 'permute'):    # Pytorch
            input_rggb = input_rggb.cpu().numpy()
        elif hasattr(input_rggb, 'shape'):      # Numpy
            pass
        else:
            raise NotImplementedError("Input must be a numpy array or pytorch tensor, current type: {}".format(type(input_rggb)))

        wb_gain = meta_data["wb_gain"][idx]
        if hasattr(wb_gain, 'permute'):    # Pytorch
            wb_gain = wb_gain.numpy()
        CCM = meta_data["ccm"][idx]
        if hasattr(CCM, 'permute'):    # Pytorch
            CCM = CCM.numpy()
    
        input_rgb = RawUtils.bayer01_2_rgb01(
            RawUtils.rggb2bayer(input_rggb), wb_gain=wb_gain, CCM=CCM, gamma = 2.2)
        input_bgr = cv2.cvtColor(input_rgb, cv2.COLOR_RGB2BGR)
        input_bgr = (input_bgr*255.0).astype(np.uint8)

        return input_bgr

if __name__ == "__main__":
    dir_pattern = "D:/image_database/SID/SID/Sony/long/*.ARW"
    device = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')
    # device = 'cpu'
    bPreLoadAll = False
    np.random.seed(42)

    dataset = TimBrooksRawDataset(dir_pattern, 1024, 1024, bPreLoadAll=False, device=device)
    train_loader = create_dataloader(dataset, 2)

    for batch_idx, (inputs_rggb_gt, inputs_rggb_noisy, inputs_rggb_variance, meta_data) in enumerate(train_loader):

        bgr888_noisy = dataset.ConvertDatasetImgToBGR888(inputs_rggb_noisy, meta_data, 0)
        cv2.imwrite("rgb_noisy.bmp", bgr888_noisy)

        bgr888_gt = dataset.ConvertDatasetImgToBGR888(inputs_rggb_gt, meta_data)
        cv2.imwrite("rgb_gt.bmp", bgr888_gt)

        break

    filename = os.path.basename(__file__) if '__file__' in globals() else os.path.basename(sys.argv[0])
    print(filename, ": Image saved as output.jpg")