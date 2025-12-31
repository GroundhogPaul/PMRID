import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

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

from PlotAgainShotRead import interpolate_gain_var, GetJin1ShotAndReadVar
from RawDataset import RawArrayToTensor, PMRIDRawDataset, create_dataloader 

class SingleNoiseRawDataset(PMRIDRawDataset):
    def __init__(self, dir_pattern, height=1024, width=1024, bPreLoadAll=False, device='cpu'):
        super().__init__(dir_pattern, height, width, bPreLoadAll=bPreLoadAll, device=device)
        self.black_level_jin1 = 64.0
        self.white_level_jin1 = 1023.0
        self.blc_01_jin1 = self.black_level_jin1 / self.white_level_jin1

    def get_noise(self, input_bayer_01, noise_type='Gaussian'):
        SensorGain = 64
        var_shot, var_read = GetJin1ShotAndReadVar(SensorGain)
        sig_shot, sig_read = torch.sqrt(torch.tensor(var_shot)), torch.sqrt(torch.tensor(var_read))

        if noise_type == 'PoissonGaussian':
            # Poisson and Gaussian noise model
            shot_noise = torch.poisson(input_bayer_01 / var_shot) * var_shot
            read_noise = torch.randn(input_bayer_01.shape) * sig_read
            noisy_01 = shot_noise + read_noise
        elif noise_type == 'Gaussian':
            # Gaussian noise model
            variance = input_bayer_01 * var_shot + var_read
            noise_01 = torch.randn_like(input_bayer_01) * torch.sqrt(variance)

        return noise_01

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
        blc_01 = float(black_level) / white_level
        B, H, W = input_bayer.shape

        # data transform          
        input_bayer = self.random_crop_and_flip(input_bayer, bayer_pattern, H_crop=self.height, W_crop=self.width, p_flip_ud=0.5, p_flip_lr=0.5)
        input_bayer_01 = input_bayer / white_level
        input_bayer_01_withOutBlc = torch.clamp(input_bayer_01 - blc_01, 0.0, 1.0) # ideal image with no blc

        # brightness and contrast augmentation
        input_bayer_AnyVal = tvtransforms.ColorJitter(brightness=(0.2, 1.2), contrast=(0.5, 1.5))(input_bayer_01_withOutBlc) # ideal image with augmentation
        clean_rggb_AnyVal = RawUtils.bayer_to_rggb(input_bayer_AnyVal, "RGGB")

        # ----- random AWB ----- #
        # red_gain = np.random.uniform(0.8, 1.2)
        # blue_gain = np.random.uniform(0.8, 1.2)
        # AWB_gain = torch.tensor([1/red_gain, 1.0, 1.0, 1/blue_gain], dtype=torch.float32).to(self.device)
        # clean_rggb_AnyVal = clean_rggb_AnyVal * AWB_gain.view(1, 1, 4)

        # ----- the clean image ----- #  # to [H/2, W/2, 4] RGGB
        clean_rggb_withoutBlc_01 = torch.clamp(clean_rggb_AnyVal, 0.0, 1.0) 
        clean_rggb_withBlc_01 = torch.clamp(clean_rggb_withoutBlc_01 + self.blc_01_jin1, 0.0, 1.0) 

        # ----- add random noise ----- 
        noise = self.get_noise(clean_rggb_withoutBlc_01) # noise from light
        noisy_rggb_withoutBlc = clean_rggb_withoutBlc_01 + noise # noisy from light and without blc

        # ----- clip the value out of [0,1] when added with blc ----- #
        noisy_rggb_withBlc = noisy_rggb_withoutBlc + self.blc_01_jin1
        noisy_rggb_withBlc_01 = torch.clamp(noisy_rggb_withBlc.to(torch.float32), 0, 1) # ----- input noisy from camera ----- #

        # save meta data
        wb_gain = ArwReaderCur.GetWBgain01("RGGB")
        ccm3x3 = ArwReaderCur.GetCCM()
        meta_data = {
            'black_level': black_level,
            'wb_gain': wb_gain,
            'ccm': ccm3x3
        }


        GT = clean_rggb_withBlc_01.permute(2,0,1).to(self.device)
        Noisy = noisy_rggb_withBlc_01.permute(2,0,1).to(self.device)
        return GT, Noisy, meta_data

    def ConvertDatasetImgToBGR888(self, inputs_rggb, meta_data, idx = 0):
        input_rggb = inputs_rggb[idx]
        input_rggb = input_rggb - self.blc_01_jin1
        input_rggb = input_rggb.permute(1, 2, 0).to('cpu')

        if hasattr(input_rggb, 'permute'):    # Pytorch
            input_rggb = input_rggb.detach().cpu().numpy()
        elif hasattr(input_rggb, 'shape'):      # Numpy
            pass
        else:
            raise NotImplementedError("Input must be a numpy array or pytorch tensor, current type: {}".format(type(input_rggb)))

        wb_gain = meta_data["wb_gain"][idx]
        if hasattr(wb_gain, 'permute'):    # Pytorch
            wb_gain = wb_gain.detach().numpy()
        CCM = meta_data["ccm"][idx]
        if hasattr(CCM, 'permute'):    # Pytorch
            CCM = CCM.detach().numpy()
    
        input_rgb = RawUtils.bayer01_2_rgb01(
            RawUtils.rggb2bayer(input_rggb), wb_gain=wb_gain, CCM=CCM, gamma = 2.2)
        input_bgr = cv2.cvtColor(input_rgb, cv2.COLOR_RGB2BGR)
        input_bgr = (input_bgr*255.0).astype(np.uint8)

        return input_bgr

if __name__ == "__main__":
    dir_pattern = "D:/image_database/SID/SID/Sony/long/*.ARW"
    device, bPreLoadAll = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu'), False
    # device = 'cpu'

    seed = 40
    torch.manual_seed(seed)
    np.random.seed(seed)

    dataset = SingleNoiseRawDataset(dir_pattern, 1024, 1024, bPreLoadAll=bPreLoadAll, device=device)
    train_loader = create_dataloader(dataset, 2)

    for batch_idx, (inputs_rggb_gt, inputs_rggb_noisy, meta_data) in enumerate(train_loader):

        bgr888_noisy = dataset.ConvertDatasetImgToBGR888(inputs_rggb_noisy, meta_data, 0)
        cv2.imwrite("rgb_noisy.bmp", bgr888_noisy)

        bgr888_gt = dataset.ConvertDatasetImgToBGR888(inputs_rggb_gt, meta_data, 0)
        cv2.imwrite("rgb_gt.bmp", bgr888_gt)

        break

    filename = os.path.basename(__file__) if '__file__' in globals() else os.path.basename(sys.argv[0])
    print(filename, ": Image saved as output.jpg")