import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)  # 插入开头，优先搜索

import utilData
import utilBasic
import rawpy
import imageio
import glob
import numpy as np
import torch
from utilRaw import RawUtils
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
from run_benchmark import KSigma, Official_Ksigma_params
import torchvision.transforms as tvtransforms
import copy
import cv2
import math
from utilBasic import print_gpu_memory_stats

class NoiseProfile:
    Noise_model = 'Gaussian'                          #'Gaussian' or 'PoissonGaussian'
    k_coefs = [0.0005995267, 0.00868861]              # k(iso) = k1 * iso + k0
    b_coefs = [7.11772e-7, 6.514934e-4, 0.11492713]   # b(iso) = b2 * iso^2 + b1 * iso + b0

class NoiseProfileFunc:
    def __init__(self, noise_profile: NoiseProfile):
        self.polyK = np.poly1d(noise_profile.k_coefs)
        self.polyB = np.poly1d(noise_profile.b_coefs)

    def __call__(self, iso):
        k = self.polyK(iso)
        b = self.polyB(iso)

        return k, b

def plot_noise_profile_curves():
    """绘制k和b关于ISO的曲线"""
    
    # 创建噪声配置文件实例
    noise_profile = NoiseProfile()
    noise_func = NoiseProfileFunc(noise_profile)
    
    # 创建ISO范围（从100到6400，覆盖常见ISO值）
    iso_range = np.linspace(100, 6400, 100)
    
    # 计算k和b值
    k_values = noise_func.polyK(iso_range)
    b_values = noise_func.polyB(iso_range)
    
    # 创建图形
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # 绘制k曲线
    ax1.plot(iso_range, k_values, 'b-', linewidth=2, label='k(ISO)')
    ax1.set_xlabel('ISO', fontsize=12)
    ax1.set_ylabel('k value', fontsize=12)
    ax1.set_title('k vs ISO', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=12)
    
    # 在k曲线上标记一些关键点
    key_isos = [100, 400, 800, 1600, 3200, 6400]
    key_k_values = noise_func.polyK(key_isos)
    ax1.scatter(key_isos, key_k_values, color='red', s=50, zorder=5)
    for iso, k_val in zip(key_isos, key_k_values):
        ax1.annotate(f'({iso}, {k_val:.4f})', 
                    (iso, k_val), 
                    xytext=(10, 10), 
                    textcoords='offset points',
                    fontsize=9,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))
    
    # 绘制b曲线
    ax2.plot(iso_range, b_values, 'r-', linewidth=2, label='b(ISO)')
    ax2.set_xlabel('ISO', fontsize=12)
    ax2.set_ylabel('b value', fontsize=12)
    ax2.set_title('b vs ISO', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=12)
    
    # 在b曲线上标记一些关键点
    key_b_values = noise_func.polyB(key_isos)
    ax2.scatter(key_isos, key_b_values, color='blue', s=50, zorder=5)
    for iso, b_val in zip(key_isos, key_b_values):
        ax2.annotate(f'({iso}, {b_val:.6f})', 
                    (iso, b_val), 
                    xytext=(10, 10), 
                    textcoords='offset points',
                    fontsize=9,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7))
    
    # 设置科学计数法显示
    ax2.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    ax2.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    
    plt.tight_layout()
    plt.show()
    
    # 打印多项式表达式
    print("=" * 50)
    print("噪声模型多项式表达式:")
    print("=" * 50)
    print(f"k(ISO) = {noise_profile.k_coefs[0]:.10f} * ISO + {noise_profile.k_coefs[1]:.8f}")
    print(f"b(ISO) = {noise_profile.b_coefs[0]:.10f} * ISO² + {noise_profile.b_coefs[1]:.8f} * ISO + {noise_profile.b_coefs[2]:.8f}")
    print()
    
    # 打印关键ISO值的计算结果
    print("关键ISO值的噪声参数:")
    print("-" * 50)
    print(f"{'ISO':>8} {'k(ISO)':>12} {'b(ISO)':>15}")
    print("-" * 50)
    for iso in key_isos:
        k_val = noise_func.polyK(iso)
        b_val = noise_func.polyB(iso)
        print(f"{iso:>8} {k_val:>12.6f} {b_val:>15.8f}")


class RawArrayToTensor:
    """[H,W] ndarray ---> [1,H,W] Tensor"""
    def __call__(self, arr):
        return torch.from_numpy(arr).float().unsqueeze(0)  # [1, H, W]

class PMRIDRawDataset(Dataset):
    def __init__(self, dir_pattern, height=1024, width=1024, bPreLoadAll=False, device='cpu'):
        self.filenames = glob.glob(dir_pattern) 
        self.height = height
        self.width = width
        self.device = device

        self.bPreLoadAll = bPreLoadAll
        if self.bPreLoadAll:
            self.preloaded_raw = []
            self.preloaded_bayer_device = []
            print("Dataset: preload all raw files to device: ", device, " begin.")
            for idx in range(len(self.filenames)):
                self.preloaded_raw.append(rawpy.imread(self.filenames[idx]))
                self.preloaded_bayer_device.append(RawArrayToTensor()(self.preloaded_raw[-1].raw_image.astype(np.float32)).to(self.device))
                if idx % 100 == 0:
                    print(f"----- preload {idx+1}/{len(self.filenames)} raw files. -----")
                    print_gpu_memory_stats(self.device)
            print("Dataset: preload all raw files to device: ", device, " end.")

    def random_crop_and_flip(self, input_bayer:np.ndarray, bayer_pattern, H_crop=1024, W_crop=1024, p_flip_ud=0.5, p_flip_lr=0.5) -> np.ndarray:
        """
        Random flip and crop a bayter-patterned image, and normalize the bayer pattern to RGGB.
        """    
        B, H, W = input_bayer.shape
        
        if np.array_equal(bayer_pattern, [[0, 1], [3, 2]]):    #'RGGB'
            crop_x_offset, crop_y_offset = 0, 0
        elif np.array_equal(bayer_pattern, [[1, 0], [2, 3]]):  #'GRBG'
            crop_x_offset, crop_y_offset = 1, 0
        elif np.array_equal(bayer_pattern, [[2, 3], [0, 1]]):  #'GBRG'
            crop_x_offset, crop_y_offset = 0, 1
        elif np.array_equal(bayer_pattern, [[2, 1], [3, 0]]):  #'BGGR'
            crop_x_offset, crop_y_offset = 1, 1

        flip_lr = np.random.rand() < p_flip_lr
        flip_ud = np.random.rand() < p_flip_ud
        if flip_lr:
            crop_x_offset = (crop_x_offset + 1) % 2
            crop_y_offset = crop_y_offset
        if flip_ud:
            crop_x_offset = crop_x_offset
            crop_y_offset = (crop_y_offset + 1) % 2

        crop_x_start = np.random.randint(0, W - W_crop)
        crop_y_start = np.random.randint(0, H - H_crop)
        crop_x_start = crop_x_start // 2 * 2 + crop_x_offset
        crop_y_start = crop_y_start // 2 * 2 + crop_y_offset

        crop_bayer = input_bayer[..., crop_y_start:crop_y_start+H_crop, crop_x_start:crop_x_start+W_crop]
        
        if flip_lr:
            crop_bayer = torch.flip(crop_bayer, dims=[2])
        if flip_ud:
            crop_bayer = torch.flip(crop_bayer, dims=[1])
        
        return crop_bayer

    def add_noise(self, input_bayer_01, noise_type='Gaussian'):
        iso = torch.randint(100, 6400, (1,))
        KSigmaCur = KSigma(Official_Ksigma_params['K_coeff'], Official_Ksigma_params['B_coeff'], Official_Ksigma_params['anchor'])

        iso = iso.item()
        k, sigma = KSigmaCur.GetKSigma(iso)
        kSigmaCalibLevel = 959 # 1023 - black_level(64), copy from run_benchmark.py
        input_bayer = input_bayer_01 * kSigmaCalibLevel # to kSigma calibrate scale 

        if noise_type == 'PoissonGaussian':
            # Poisson and Gaussian noise model
            shot_noise = torch.poisson(input_bayer / k) * k
            read_noise = torch.randn(input_bayer.shape) * torch.sqrt(torch.tensor(sigma))
            noisy_bayer = shot_noise + read_noise
        
        elif noise_type == 'Gaussian':
            # Gaussian noise model
            variance = input_bayer * k + sigma
            noise = torch.randn_like(input_bayer) * torch.sqrt(variance)
            noisy_bayer = input_bayer + noise

        else:
            noisy_bayer = input_bayer

        noisy_bayer_01 = noisy_bayer / kSigmaCalibLevel  # to original scale

        return torch.clamp(noisy_bayer_01.to(torch.float32), 0, 1), copy.deepcopy(KSigmaCur)
    
    def __getitem__(self, index):
        # input_raw = rawpy.imread("D:/image_database/SID/SID/Sony/long/00002_00_10s.ARW")
        # read raw
        if self.bPreLoadAll:
            input_raw = self.preloaded_raw[index]
            input_bayer = self.preloaded_bayer_device[index]
        else:
            input_raw = rawpy.imread(self.filenames[index])
            input_bayer = input_raw.raw_image.astype(np.float32)
            input_bayer = np.ascontiguousarray(input_bayer)
            input_bayer = RawArrayToTensor()(input_bayer)
            input_bayer = input_bayer.to(self.device)
        bayer_pattern = input_raw.raw_pattern
        white_level = input_raw.white_level
        black_level = input_raw.black_level_per_channel[0]
        B, H, W = input_bayer.shape

        wb_gain = np.array([wb / 1024.0 for wb in input_raw.camera_whitebalance]),
        ccm = input_raw.rgb_xyz_matrix[:3, :]

        # ---------- data transform ---------- #         
        input_bayer = self.random_crop_and_flip(input_bayer, bayer_pattern, H_crop=self.height, W_crop=self.width, p_flip_ud=0.5, p_flip_lr=0.5)

        # !!!!! TODO: the black level + jitter + noise is so wrong (with black level) !!!!!
        # ----- black level and normalize ----- #
        input_bayer_01 = input_bayer / white_level  # no black_level at all, copied from benchmark.py.BenchmarkLoader
        # input_bayer_01 = (input_bayer - black_level) / (white_level - black_level)

        # ----- brightness and contrast augmentation ----- #
        print(input_bayer_01.min(), input_bayer_01.max(), input_bayer_01.mean())
        # input_bayer_01 = tvtransforms.ColorJitter(brightness=(0.2, 1.2), contrast=(0.5, 1.5))(input_bayer_01)
        input_bayer_01 = tvtransforms.ColorJitter(brightness=(1.2, 1.2), contrast=(1.5, 1.5))(input_bayer_01)
        print(input_bayer_01.min(), input_bayer_01.max(), input_bayer_01.mean())
        input_bayer_01 = torch.clamp(input_bayer_01, 0.0, 1.0)
        input_rggb_01 = RawUtils.bayer_to_rggb(input_bayer_01, "RGGB")  # to [H/2, W/2, 4] RGGB

        # add random noise
        input_rggb_01_noisy, kSigmaCur = self.add_noise(input_rggb_01)

        # apply k sigma transform
        input_rggb_01_noisy_k = kSigmaCur(input_rggb_01_noisy, iso=kSigmaCur.iso_last, inverse=False)

        # save meta data
        meta_data = {
            'iso': kSigmaCur.iso_last, 
            'black_level': input_raw.black_level_per_channel[0],
            'wb_gain': wb_gain,
            'ccm': ccm
        }

        return input_rggb_01, input_rggb_01_noisy, input_rggb_01_noisy_k, meta_data

    def __len__(self):
        return len(self.filenames)


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
            input_raw = self.preloaded_raw[index]
            input_bayer = self.preloaded_bayer_device[index]
        else:
            input_raw = rawpy.imread(self.filenames[index])
            # input_raw = rawpy.imread("D:/image_database/SID/SID/Sony/long/00002_00_10s.ARW") # test code
            input_bayer = input_raw.raw_image.astype(np.float32)
            input_bayer = np.ascontiguousarray(input_bayer)
            input_bayer = RawArrayToTensor()(input_bayer)
            input_bayer = input_bayer.to(self.device)
        bayer_pattern = input_raw.raw_pattern
        white_level = input_raw.white_level
        black_level = input_raw.black_level_per_channel[0]
        B, H, W = input_bayer.shape

        wb_gain = np.array([wb / 1024.0 for wb in input_raw.camera_whitebalance]),
        ccm = input_raw.rgb_xyz_matrix[:3, :]

        # data transform          
        input_bayer = self.random_crop_and_flip(input_bayer, bayer_pattern, H_crop=self.height, W_crop=self.width, p_flip_ud=0.5, p_flip_lr=0.5)
        input_bayer_01 = input_bayer / white_level  # no black_level at all, copied from benchmark.py.BenchmarkLoader
        # input_bayer_01 = (input_bayer - black_level) / (white_level - black_level)
        # input_bayer_01 = (input_bayer - black_level) / white_level

        # brightness and contrast augmentation
        input_bayer_01 = tvtransforms.ColorJitter(brightness=(0.2, 1.2), contrast=(0.5, 1.5))(input_bayer_01)
        input_bayer_01 = torch.clamp(input_bayer_01, 0.0, 1.0)
        input_rggb_01 = RawUtils.bayer_to_rggb(input_bayer_01, "RGGB")  # to [H/2, W/2, 4] RGGB

        # add random noise
        input_rggb_01_noisy, shot_noise, read_noise = self.add_noise(input_rggb_01)
        input_bayer_01_noisy = RawUtils.rggb2bayer(input_rggb_01_noisy)
        # print("\n train set: shot_noise:", shot_noise.item(), " read_noise:", read_noise.item())
        input_rggb_variance = torch.sqrt(input_rggb_01_noisy * shot_noise + read_noise)

        # save meta data
        meta_data = {
            'black_level': input_raw.black_level_per_channel[0],
            'wb_gain': wb_gain,
            'ccm': ccm
        }

        return input_rggb_01, input_rggb_01_noisy, input_rggb_variance, meta_data

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
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=False,
        drop_last=True  # To ensure consistent batch sizes
    )


if __name__ == "__main__":
    dir_pattern = "D:/image_database/SID/SID/Sony/long/*.ARW"
    # device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
    device = 'cpu'
    bPreLoadAll = False
    np.random.seed(42)
    # dataset = PMRIDRawDataset(dir_pattern, device=device, bPreLoadAll=bPreLoadAll)
    dataset = TimBrooksRawDataset(dir_pattern, device=device, bPreLoadAll=bPreLoadAll)
    print(f"number of clean raw images for training:{len(dataset)}")
    input_rggb_01, input_rggb_01_noisy, input_rggb_01_noisy_k, meta_data = dataset[0]
    # input_rggb_01_denoise = (input_rggb_01_noisy_k - meta_data['cvt_b']) / meta_data['cvt_k']
    # print(f'cvt_b:{meta_data["cvt_b"]}, cvt_k:{meta_data["cvt_k"]}')

    input_rggb_01 = input_rggb_01.cpu().numpy()
    input_rggb_01_noisy = input_rggb_01_noisy.cpu().numpy()
    
    wb_gain = meta_data["wb_gain"][0]
    wb_gain = wb_gain[[0,1,2]]
    CCM = meta_data["ccm"]
    
    rgb_clean = RawUtils.bayer01_2_rgb01(
        RawUtils.rggb2bayer(input_rggb_01), wb_gain=wb_gain, CCM=CCM, gamma = 2.2)
    rgb_noisy = RawUtils.bayer01_2_rgb01(
        RawUtils.rggb2bayer(input_rggb_01_noisy), wb_gain=wb_gain, CCM=CCM, gamma=2.2)
    

    # 保存为 JPEG 图像
    imageio.imwrite('rgb_clean.bmp', (rgb_clean * 255).astype(np.uint8))
    imageio.imwrite('rgb_noisy.bmp', (rgb_noisy * 255).astype(np.uint8))

    print("Image saved as output.jpg")
    
    # plot_noise_profile_curves()
    # input_raw.raw_image = np.uint16(input_rggb_noisy * 16383)
    # rgb_image = input_raw.postprocess()
    # imageio.imwrite('test.jpg', rgb_image)
    # print("finish")
