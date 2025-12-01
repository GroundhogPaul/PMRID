import utilBasicRun
from models.net_torch import *

import numpy as np
import os
import cv2
from utilRaw import RawUtils
from utilVrf import read_vrf, save_vrf_image, save_raw_image

# ---------- read vrf ---------- #
# ----- assert paths ----- #
sFolder = r"D:\image_database\jn1_mfnr_bestshot\unpacked"
assert os.path.exists(sFolder), f"Data folder does not exist: {sFolder}"
# sVrfFile = "53/0_unpacked.vrf"
sVrfFile = "33/5_unpacked.vrf"
sVrfPath = os.path.join(sFolder, sVrfFile)
assert os.path.exists(sVrfPath), f"VRF file does not exist: {sVrfPath}"

# ----- read vrf ----- #
W, H = 4080, 3060
black_level = 64
white_level = 1023
dgain = 1.0

bayer01_GRBG_noisy = read_vrf(sVrfPath, W, H, black_level, dgain, white_level)
bayer01_RGGB_noisy = np.fliplr(bayer01_GRBG_noisy)
bayer01_BGGR_noisy = RawUtils.rggb2bggr(bayer01_RGGB_noisy)

bgr01_noisy = RawUtils.bayer01_2_rgb01(bayer01_BGGR_noisy, wb_gain=[1.5156, 1.0, 1.7421], CCM=np.eye(3))
bgr_noisy = (bgr01_noisy*255.0).astype(np.uint8)
cv2.imwrite("noisy_rgb.png", bgr_noisy)

# ---------- Denoise ---------- #
class KSigmaCur:

    def __init__(self, V: float = 959.0):
        self.V = V

    def __call__(self, img_01, iso: float, inverse=False):
        k, sigma = 3.84565949, 33.43866601  # this should be calculate from again and calib,
        # but we just use fixed value here for simplicity

        k_a, sigma_a = 0.96793133, 2.97945289

        cvt_k = k_a / k
        cvt_b = (sigma / (k ** 2) - sigma_a / (k_a ** 2)) * k_a

        img = img_01 * self.V

        if not inverse:
            img = img * cvt_k + cvt_b
        else:
            img = (img - cvt_b) / cvt_k

        return img / self.V


from models.net_torch import NetworkPMRID as Network
class Denoiser:

    def __init__(self, model_path, ksigma: KSigmaCur, inp_scale=256.0):
        net = Network()
        net.load_CKPT(str(model_path), device=torch.device('cpu'))
        net.eval()

        self.net = net
        self.ksigma = ksigma
        self.inp_scale = inp_scale

    def pre_process(self, bayer_01: np.ndarray):
        rggb = RawUtils.bayer2rggb(bayer_01)
        rggb = rggb.clip(0, 1)

        H, W = rggb.shape[:2]
        ph, pw = (32-(H % 32))//2, (32-(W % 32))//2
        rggb = np.pad(rggb, [(ph, ph), (pw, pw), (0, 0)], 'constant')
        inp_rggb = rggb.transpose(2, 0, 1)[np.newaxis]
        self.ph, self.pw = ph, pw
        return inp_rggb

    def run(self, bayer_01: np.ndarray, iso: float):
        inp_rggb_01 = self.pre_process(bayer_01)
        inp_rggb_ksigma = self.ksigma(inp_rggb_01, iso)
        inp_rggb = inp_rggb_ksigma * self.inp_scale
        print("inp_rggb: ", inp_rggb.min(), inp_rggb.max())

        inp = np.ascontiguousarray(inp_rggb)
        input_tensor = torch.from_numpy(inp).float()
        pred = self.net(input_tensor)[0] / self.inp_scale

        # import ipdb; ipdb.set_trace()
        pred = pred.detach().cpu().numpy()
        pred = pred.transpose(1, 2, 0)
        pred = self.ksigma(pred, iso, inverse=True)

        ph, pw = self.ph, self.pw
        pred = pred[ph:-ph, pw:-pw]
        return RawUtils.rggb2bayer(pred)

kSigma = KSigmaCur()

model_path =  "D:/users/xiaoyaopan/PxyAI/PMRID_OFFICIAL/PMRID/models/torch_pretrained.ckp"
Denoiser = Denoiser(model_path, kSigma)

bayer01_RGGB_denoise = Denoiser.run(bayer01_RGGB_noisy, iso=1600.0)
bayer01_RGGB_denoise = np.clip(bayer01_RGGB_denoise, 0.0, 1.0)
import skimage

# ---------- save image ---------- #
# ----- save png ----- #
bayer01_BGGR_denoise = RawUtils.rggb2bggr(bayer01_RGGB_denoise)

bgr01_denoise = RawUtils.bayer01_2_rgb01(bayer01_BGGR_denoise, wb_gain=[1.5156, 1.0, 1.7421], CCM=np.eye(3))
bgr_denoise = (bgr01_denoise*255.0).astype(np.uint8)
cv2.imwrite("denoise_rgb.bmp", bgr_denoise)

import skimage
psnr = skimage.metrics.peak_signal_noise_ratio(bgr_denoise, bgr_noisy)
print("psnr_bgr = ", psnr)
print(bayer01_RGGB_denoise.min(), bayer01_RGGB_denoise.max())
print(bayer01_RGGB_noisy.min(), bayer01_RGGB_noisy.max())
psnr = skimage.metrics.peak_signal_noise_ratio(bayer01_RGGB_denoise, bayer01_RGGB_noisy) 
print("psnr_bayer01 = ", psnr)

bgr_denoise_std = cv2.imread("denoise_rgb_std.bmp")
errNorm2 = np.linalg.norm(bgr_denoise.astype(np.float32) - bgr_denoise_std.astype(np.float32))
# print(errNorm2)

# ----- save vrf ----- #
out_ratio = 4  #out 12bit
out_black_level = black_level * out_ratio  # 根据实际情况调整
out_white_level = (white_level + 1) * out_ratio - 1
bayer01_GRBG_denoise = np.fliplr(bayer01_RGGB_denoise)
denoised_image = save_raw_image(bayer01_GRBG_denoise, os.path.join("", '53.raw'), out_white_level, out_black_level)
save_vrf_image(denoised_image, sVrfPath, os.path.join("", '53.vrf'), out_white_level)