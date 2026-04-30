import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)
import utilData

import glob
import numpy as np
import torch
from RawContainer.utilRaw import RawUtils 
from RawContainer.utilVrf import vrf, CFAPatternEnum, CFAPatternEnum
from torch.utils.data import Dataset
import torch.nn.functional as F

import cv2

from ImgUnprocess import unprocess
from ImgProcess import process_meta_data

class NumpyHW2TorchChHW:
    """[H,W] ndarray ---> [1,H,W] Tensor"""
    def __call__(self, arr):
        return torch.from_numpy(arr).float().unsqueeze(0)  # [1, H, W]

class CropDatasetBasic(Dataset):
    def __init__(self, lstImgPattern, Hbayer, Wbayer, device = 'cpu'):
        assert Hbayer % 2 == 0
        assert Wbayer % 2 == 0
        self.Hbayer = Hbayer
        self.Wbayer = Wbayer
        self.Hrggb = Hbayer // 2
        self.Wrggb = Wbayer // 2
        self.device = device

        self.collectImgPath(lstImgPattern)
        
    def collectImgPath(self, lstImgPattern):
        if isinstance(lstImgPattern, str):
            lstImgPattern = [lstImgPattern]
        else:
            assert isinstance(lstImgPattern, list), "lstImgPattern should be a string or a list of strings"

        self.imgPaths = []
        for pattern in lstImgPattern:
            imgPaths = glob.glob(pattern)
            if not imgPaths:
                raise AssertionError(f"Folder '{pattern}' 没有找到文件")
            self.imgPaths.extend(imgPaths)
        print("Number of image paths collected:", len(self.imgPaths))

    def __getitem__(self, index):
        raise NotImplementedError("CropDatasetBasic is an abstract class, please implement __getitem__ method in the subclass")
    
    def __len__(self):
        return len(self.imgPaths)

class CropDatasetVrf(CropDatasetBasic):
    def __init__(self, lstImgPattern, Hbayer, Wbayer, device = 'cpu'):
        super().__init__(lstImgPattern, Hbayer, Wbayer, device=device)
    
    def random_crop_and_flip(self, input_bayer:np.ndarray, bayer_pattern_enum, H_crop=1024, W_crop=1024, p_flip_ud=0.5, p_flip_lr=0.5, p_transpose=0.5) -> np.ndarray:
        # Random flip and crop a bayter-patterned image, and normalize the bayer pattern to RGGB.
        if len(input_bayer.shape) == 2:
            H, W = input_bayer.shape
        elif len(input_bayer.shape) == 3:
            B, H, W = input_bayer.shape
        else:
            AssertionError
        
        # crop_x_offset, crop_y_offset = 0, 0
        if bayer_pattern_enum == CFAPatternEnum.RGGB:
            crop_x_offset, crop_y_offset = 0, 0
        elif bayer_pattern_enum == CFAPatternEnum.GRBG:
            crop_x_offset, crop_y_offset = 1, 0
        elif bayer_pattern_enum == CFAPatternEnum.GBRG:
            crop_x_offset, crop_y_offset = 0, 1
        elif bayer_pattern_enum == CFAPatternEnum.BGGR:
            crop_x_offset, crop_y_offset = 1, 1
        elif bayer_pattern_enum == CFAPatternEnum.MONO:
            crop_x_offset, crop_y_offset = 0, 0
        else:
            assert False, "bayer_pattern not supported: {}".format(bayer_pattern_enum)

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
            crop_bayer = NumpyHW2TorchChHW()(crop_bayer)
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

    def __getitem__(self, index):
        vrfCur = vrf(self.imgPaths[index])
        bayer_in = vrfCur.get_raw_image()
        bayer_pattern_enum = vrfCur.get_raw_pattern_num()
        bayer_crop_RGGB = self.random_crop_and_flip(
            bayer_in, bayer_pattern_enum, H_crop=self.Hbayer, W_crop=self.Wbayer, p_flip_ud=0.5, p_flip_lr=0.5)
        bayer_crop_RGGB = bayer_crop_RGGB.to(self.device)
        white_level = vrfCur.get_white_level()
        black_level = vrfCur.get_black_level()
        wb_gain = vrfCur.get_WBgain_01(CFAPatternEnum.RGGB).astype(np.float32) # after random flip, the pattern is always RGGB
        ccm3x3 = vrfCur.get_CCM().astype(np.float32)

        meta_data = {
            'white_level': white_level,
            'black_level': black_level,
            'wb_gain': wb_gain,
            'ccm3x3': ccm3x3
        }

        rggb_crop = RawUtils.bayer_to_rggb(bayer_crop_RGGB, "RGGB")  # to [H/2, W/2, 4] RGGB
        rggb_crop = (rggb_crop.float() - black_level) / white_level  # to [0, 1]

        rggb_crop = torch.clamp(rggb_crop, 0, 1)

        return rggb_crop, meta_data
    
    def rggb01_2_bgr888(self, rggb, meta_data):
        wb_gain = meta_data['wb_gain']
        CCM = meta_data['ccm3x3']
        rgb = RawUtils.bayer01_2_rgb01(
            RawUtils.rggb2bayer(rggb), wb_gain=wb_gain, CCM=CCM, gamma = 2.2)
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        bgr888 = (bgr*255.0).astype(np.uint8)

        return bgr888

class CropDatasetJpg(CropDatasetBasic):
    def __init__(self, lstImgPattern, Hbayer, Wbayer, device = 'cpu'):
        super().__init__(lstImgPattern, Hbayer, Wbayer, device=device)
    
    def __getitem__(self, index):
        img_path = self.imgPaths[index]
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = torch.from_numpy(img).to(self.device)

        # ----- pad to bigger than net work size ----- //
        h, w, c = img.shape
        pad_h = max(0, self.Hrggb - h)
        pad_w = max(0, self.Wrggb - w)

        if pad_h > 0 or pad_w > 0:
            img = F.pad(img, (0, 0, 0, pad_w, 0, pad_h), mode = 'constant', value=127)

        # ----- crop to network size ----- #
        h, w, c = img.shape   # 此时 h >= self.Hrggb, w >= self.Wrggb
        top = torch.randint(0, h - self.Hrggb + 1, (1,)).item()
        left = torch.randint(0, w - self.Wrggb + 1, (1,)).item()
        img = img[top:top + self.Hrggb, left:left + self.Wrggb, :]
        
        # ----- flip -----
        if torch.rand(1) < 0.5:
            img = torch.flip(img, dims=[-2])  # 倒数第二维是宽度W
        if torch.rand(1) < 0.5:
            img = torch.flip(img, dims=[-3])  # 倒数第三维是高度H
        if torch.rand(1) < 0.5:
            dim_order = list(range(img.dim()))
            dim_order[-3], dim_order[-2] = dim_order[-2], dim_order[-3]
            img = img.permute(dim_order)
        
        img, meta_data = unprocess(img)

        return img, meta_data

    def rggb01_2_bgr888(self, rggb, meta_data, bCPU=True):
        assert isinstance(rggb, torch.Tensor)
        bgr888 = process_meta_data(rggb, meta_data)
        if bCPU:
            bgr888 = bgr888.cpu().numpy()
        return bgr888
        
if __name__ == "__main__":
    seed = 39
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')

    # --------- test vrf ---------- #
    # dir_pattern = "D:/image_database/SID/SID/Sony/longVRFmini/*.vrf"
    # dataset = CropDatasetVrf(dir_pattern, Hbayer=1024, Wbayer=1024, device=device)
    # input_rggb, meta_data = dataset[0]
    # bgr888 = dataset.rggb01_2_bgr888(input_rggb, meta_data)
    # cv2.imwrite("test_CropDatasetVrf.jpg", bgr888)

    # --------- test jpg ---------- #
    dir_pattern = "D:/image_database/mirflickr25k/mirflickr/*.jpg"
    dataset = CropDatasetJpg(dir_pattern, Hbayer=1024, Wbayer=1024, device=device)
    input_rggb, meta_data = dataset[1]
    bgr888 = dataset.rggb01_2_bgr888(input_rggb, meta_data, bCPU = True)
    cv2.imwrite("test_CropDatasetJpg.jpg", bgr888)