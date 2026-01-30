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
# from utilArw import ArwReader
from utilVrf import vrf, CFAPatternEnum, read_vrf, save_vrf_image, save_raw_image 
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
from utils.KSigma import KSigma, Official_Ksigma_params
import torchvision.transforms as tvtransforms
import copy
import cv2
import math
import time
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
    def __init__(self, dir_patterns, height=1024, width=1024, bPreLoadAll=False, device='cpu'):
        if isinstance(dir_patterns, str):
            dir_patterns = [dir_patterns]
    
        self.filenames = []
        for pattern in dir_patterns:
            files = glob.glob(pattern)
            if not files:
                AssertionError(f"Folder '{pattern}' 没有找到文件")
            self.filenames.extend(files)
        self.height = height
        self.width = width
        self.device = device

        self.bPreLoadAll = bPreLoadAll
        if self.bPreLoadAll:
            self.preloaded_raw = []
            self.preloaded_bayer_device = []
            print("Dataset: preload all raw files to device: ", device, " begin.")
            for idx in range(len(self.filenames)):
                self.preloaded_raw.append(vrf(self.filenames[idx]))
                self.preloaded_bayer_device.append(RawArrayToTensor()(self.preloaded_raw[-1].get_raw_image()).to(self.device))
                self.preloaded_raw[-1].m_raw = None
                if idx % 100 == 0:
                    print(f"----- preload {idx+1}/{len(self.filenames)} raw files. -----")
                    print_gpu_memory_stats(self.device)
            print("Dataset: preload all raw files to device: ", device, " end.")

    def random_crop_and_flip(self, input_bayer:np.ndarray, bayer_pattern, H_crop=1024, W_crop=1024, p_flip_ud=0.5, p_flip_lr=0.5, p_transpose=0.5) -> np.ndarray:
        """
        Random flip and crop a bayter-patterned image, and normalize the bayer pattern to RGGB.
        """    
        if len(input_bayer.shape) == 2:
            H, W = input_bayer.shape
        elif len(input_bayer.shape) == 3:
            B, H, W = input_bayer.shape
        else:
            AssertionError
        
        # crop_x_offset, crop_y_offset = 0, 0
        if np.array_equal(bayer_pattern, [[0, 1], [3, 2]]):    #'RGGB' # TODO: this branch condition is wrong, should judged by RGGB instead of bayer pattern index
            crop_x_offset, crop_y_offset = 0, 0
        elif np.array_equal(bayer_pattern, [[1, 0], [2, 3]]):  #'GRBG'
            crop_x_offset, crop_y_offset = 1, 0
        elif np.array_equal(bayer_pattern, [[2, 3], [0, 1]]):  #'GBRG'
            crop_x_offset, crop_y_offset = 0, 1
        elif np.array_equal(bayer_pattern, [[2, 1], [3, 0]]):  #'BGGR'
            crop_x_offset, crop_y_offset = 1, 1
        else:
            assert False, "bayer_pattern not supported: {}".format(bayer_pattern)


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
        
        # ----- go to torch ---- #
        if not hasattr(crop_bayer, 'permute'):    # Numpy
            crop_bayer = np.ascontiguousarray(crop_bayer)
            crop_bayer = RawArrayToTensor()(crop_bayer)
        else:
            crop_bayer = crop_bayer.contiguous()

        if flip_lr:
            crop_bayer = torch.flip(crop_bayer, dims=[2])
        if flip_ud:
            crop_bayer = torch.flip(crop_bayer, dims=[1])
        
        bTranspose = np.random.rand() < p_transpose
        if bTranspose:
            crop_bayer = crop_bayer.permute(0,2,1)

        return crop_bayer

    def add_noise(self, input_bayer_01, noise_type='Gaussian', blcClip = 0):
        iso = torch.randint(100, 9600, (1,))
        # KSigmaCur = KSigma(Official_Ksigma_params['K_coeff'], Official_Ksigma_params['B_coeff'], Official_Ksigma_params['anchor'])
        KSigmaCur = KSigma(
            Official_Ksigma_params['K_coeff'], 
            Official_Ksigma_params['B_coeff'],
            Official_Ksigma_params['anchor'], 
            # k = 0.00251* 1023, # LuoWen param under ISO 6400
            # sigma = 1.265e-05 * 1023 * 1023
            )

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
            input_bayer = input_bayer
            variance = input_bayer * k + sigma
            noise = torch.randn_like(input_bayer) * torch.sqrt(variance)
            noisy_bayer = input_bayer + noise

        else:
            noisy_bayer = input_bayer

        noisy_bayer_01 = noisy_bayer / kSigmaCalibLevel  # to original scale

        return torch.clamp(noisy_bayer_01.to(torch.float32), blcClip, 1), copy.deepcopy(KSigmaCur)
    
    def __getitem__(self, index):
        self.max_retries = 1
        retry_count = 0
        while retry_count < self.max_retries:
            try:
                # read raw
                if self.bPreLoadAll:
                    ArwReaderCur = self.preloaded_raw[index]
                    input_bayer = self.preloaded_bayer_device[index] # GPU buffer
                    bayer_pattern = ArwReaderCur.get_raw_pattern_arr()
                    input_bayer = self.random_crop_and_flip(input_bayer, bayer_pattern, H_crop=self.height, W_crop=self.width, p_flip_ud=0.5, p_flip_lr=0.5)
                    white_level = ArwReaderCur.get_white_level()
                    black_level = ArwReaderCur.get_black_level()
                    wb_gain = ArwReaderCur.get_WBgain_01(CFAPatternEnum.RGGB).astype(np.float32) # after random flip, the pattern is always RGGB
                    ccm3x3 = ArwReaderCur.get_CCM().astype(np.float32)
                    break
                else:
                    ArwReaderCur = vrf(self.filenames[index])
                    input_bayer = ArwReaderCur.get_raw_image()
                    bayer_pattern = ArwReaderCur.get_raw_pattern_arr()
                    input_bayer = self.random_crop_and_flip(input_bayer, bayer_pattern, H_crop=self.height, W_crop=self.width, p_flip_ud=0.5, p_flip_lr=0.5)
                    input_bayer = input_bayer.to(self.device)
                    white_level = ArwReaderCur.get_white_level()
                    black_level = ArwReaderCur.get_black_level()
                    wb_gain = ArwReaderCur.get_WBgain_01(CFAPatternEnum.RGGB).astype(np.float32) # after random flip, the pattern is always RGGB
                    ccm3x3 = ArwReaderCur.get_CCM().astype(np.float32)
                    del ArwReaderCur
                    break
            except (MemoryError, np.core._exceptions._ArrayMemoryError, 
                    SystemError, OSError, Exception) as e:
                retry_count += 1
                print(f"尝试 {retry_count}/{self.max_retries} 失败: {e}")
                wait_time = 1.0
                time.sleep(wait_time)
                
                # 如果是最后尝试，抛出详细错误
                if retry_count >= self.max_retries:
                    print(f"经过 {self.max_retries} 次尝试后仍然失败")
                    print(f"错误文件: {self.filenames[index]}")
                    raise MemoryError(f"无法加载图像 {self.filenames[index]} 经过 {self.max_retries}次尝试")
            except Exception as e:
                raise e

        # B, H, W = input_bayer.shape

        # ---------- data transform ---------- #         

        # !!!!! TODO: the black level + jitter + noise is so wrong (with black level) !!!!!
        # ----- black level and normalize ----- #
        # input_bayer_01 = input_bayer / white_level  # no black_level at all, copied from benchmark.py.BenchmarkLoader
        input_bayer_01 = torch.clamp((input_bayer - black_level) / (white_level), 0.0, 1.0)

        # ----- brightness and contrast augmentation ----- #
        input_bayer_01 = tvtransforms.ColorJitter(brightness=(0.2, 1.2), contrast=(1.0, 1.0))(input_bayer_01)
        
        input_bayer_01 = torch.clamp(input_bayer_01, 0.0, 1.0)
        input_rggb_01 = RawUtils.bayer_to_rggb(input_bayer_01, "RGGB")  # to [H/2, W/2, 4] RGGB

        # add random noise
        input_rggb_01_noisy, kSigmaCur = self.add_noise(input_rggb_01, blcClip=0)

        # apply k sigma transform
        input_rggb_01_noisy_k = kSigmaCur(input_rggb_01_noisy, iso=kSigmaCur.iso_last, inverse=False)

        # save meta data
        meta_data = {
            'iso': kSigmaCur.iso_last, 
            'black_level': black_level,
            'wb_gain': wb_gain,
            'ccm': ccm3x3
        }

        # permute to torch net format
        GT = input_rggb_01
        GT = GT.permute(2,0,1).to(self.device)

        Noisy = input_rggb_01_noisy
        Noisy = Noisy.permute(2,0,1).to(self.device)

        Noisy_Ksigma = input_rggb_01_noisy_k
        Noisy_Ksigma = Noisy_Ksigma.permute(2,0,1).to(self.device)

        return GT, Noisy, Noisy_Ksigma, meta_data

    def __len__(self):
        return len(self.filenames)

    @classmethod
    def ConvertDatasetImgToBGR888(cls, inputs_rggb, meta_data, bKsigma = False, idx = 0):

        input_rggb = inputs_rggb[idx]
        input_rggb = input_rggb
        input_rggb = input_rggb.permute(1, 2, 0).to('cpu')

        if bKsigma:
            kSigmaCur = KSigma(Official_Ksigma_params['K_coeff'], Official_Ksigma_params['B_coeff'], Official_Ksigma_params['anchor'])
            iso = meta_data['iso'][idx]
            input_rggb = kSigmaCur(input_rggb, iso, inverse=True)

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
        
        # timeout=100000
    )

if __name__ == "__main__":
    # dir_pattern, bPreLoadAll, device = "D:/image_database/SID/SID/Sony/long_test/*.ARW", True, 'cuda:2'
    # dir_pattern, bPreLoadAll, device = "D:/users/xiaoyaopan/PxyAI/DataSet/Raw/Wholy/*.ARW", False, 'cpu'
    dir_pattern, bPreLoadAll, device = "D:/image_database/SID/SID/Sony/longVRFmini/*.vrf", True, 'cuda:2'
    device = torch.device(device if torch.cuda.is_available() else 'cpu')

    seed = 39
    torch.manual_seed(seed)
    np.random.seed(seed)

    dataset = PMRIDRawDataset(dir_pattern, device=device, bPreLoadAll=bPreLoadAll)
    train_loader = create_dataloader(dataset, 1, num_workers=0)

    for batch_idx, (inputs_rggb_01_gt, inputs_rggb_01_noisy, inputs_rggb_01_noisy_k, meta_data) in enumerate(train_loader):
        bgr888_gt = dataset.ConvertDatasetImgToBGR888(inputs_rggb_01_gt, meta_data, idx = 0)
        cv2.imwrite(f"rgb_gt_{meta_data['iso'][0]:.2f}.bmp", bgr888_gt)

        bgr888_noisy = dataset.ConvertDatasetImgToBGR888(inputs_rggb_01_noisy, meta_data, idx = 0)
        cv2.imwrite(f"rgb_noisy_{meta_data['iso'][0]:.2f}.bmp", bgr888_noisy)

        bgr888_noisy_k = dataset.ConvertDatasetImgToBGR888(inputs_rggb_01_noisy_k, meta_data, bKsigma=True, idx = 0)
        cv2.imwrite(f"rgb_noisy_k_{meta_data['iso'][0]:.2f}.bmp", bgr888_noisy_k)

        break
    print("Image saved as output.jpg")