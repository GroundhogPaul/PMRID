#!/usr/bin/env python3
import argparse
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
import skimage.metrics
from tqdm import tqdm

import torch
from models.net_torch import NetworkPMRID as Network
import utilBasic
from utilRaw import RawUtils
from benchmark import BenchmarkLoader, RawMeta


class KSigma:

    def __init__(self, K_coeff: Tuple[float, float], B_coeff: Tuple[float, float, float], anchor: float, V: float = 959.0):
        self.K = np.poly1d(K_coeff)
        self.Sigma = np.poly1d(B_coeff)
        self.anchor = anchor
        self.V = V
    
    def __call__(self, img_01, iso: float, inverse=False):
        k, sigma = self.K(iso), self.Sigma(iso)
        k_a, sigma_a = self.K(self.anchor), self.Sigma(self.anchor)

        cvt_k = k_a / k
        cvt_b = (sigma / (k ** 2) - sigma_a / (k_a ** 2)) * k_a

        img = img_01 * self.V

        if not inverse:
            img = img * cvt_k + cvt_b
        else:
            img = (img - cvt_b) / cvt_k

        return img / self.V

    def GetKSigma(self, iso: float):
        k = self.K(iso)
        Sigma = self.Sigma(iso)

        return k, Sigma

class Denoiser:

    def __init__(self, model_path: Path, ksigma: KSigma, inp_scale=256.0):
        net = Network()
        net.load_CKPT(str(model_path), device=torch.device('cpu'))
        net.eval()

        self.net = net
        self.ksigma = ksigma
        self.inp_scale = inp_scale

    def pre_process(self, bayer_01: np.ndarray): # 1. bayer to rggb; 2. pad to 32 multiple; 3. HWCh to BChHW
        rggb01_HWCh = RawUtils.bayer2rggb(bayer_01)
        rggb01_HWCh = rggb01_HWCh.clip(0, 1)

        H, W = rggb01_HWCh.shape[:2]
        ph, pw = (32-(H % 32))//2, (32-(W % 32))//2
        rggb01_HWCh = np.pad(rggb01_HWCh, [(ph, ph), (pw, pw), (0, 0)], 'constant')
        self.ph, self.pw = ph, pw
        rggb01_BChHW = rggb01_HWCh.transpose(2, 0, 1)[np.newaxis]
        return rggb01_BChHW
    
    def post_process(self, pred_BChHW: np.ndarray): # 1. BChHW to HWCh; 2. unpad; 3. rggb to bayer
        assert pred_BChHW.ndim == 4 and pred_BChHW.shape[0] == 1
        pred_HWCh = pred_BChHW[0].transpose(1, 2, 0)
        ph, pw = self.ph, self.pw
        pred_HWCh = pred_HWCh[ph:-ph, pw:-pw]
        return RawUtils.rggb2bayer(pred_HWCh)

    def run(self, bayer_01: np.ndarray, iso: float):
        rggb01_BChHW = self.pre_process(bayer_01)
        rggb01_BChHW_ksigma = self.ksigma(rggb01_BChHW, iso)
        rggb_BChHW_ksigma = rggb01_BChHW_ksigma * self.inp_scale

        inp = np.ascontiguousarray(rggb_BChHW_ksigma)
        input_tensor = torch.from_numpy(inp).float()
        pred_ChHW = self.net(input_tensor)[0] / self.inp_scale

        # import ipdb; ipdb.set_trace()
        pred_ChHW = pred_ChHW.detach().cpu().numpy()
        pred_HWCh = pred_ChHW.transpose(1, 2, 0)
        pred_HWCh = self.ksigma(pred_HWCh, iso, inverse=True)

        ph, pw = self.ph, self.pw
        pred_HWCh = pred_HWCh[ph:-ph, pw:-pw]
        return RawUtils.rggb2bayer(pred_HWCh)


def run_benchmark(model_path, bm_loader: BenchmarkLoader):

    ksigma = KSigma(
        K_coeff=[0.0005995267, 0.00868861],
        B_coeff=[7.11772e-7, 6.514934e-4, 0.11492713],
        anchor=1600,
    )
    denoiser = Denoiser(model_path, ksigma)

    PSNRs_rgb_denoise, SSIMs_rgb = [], []
    PSNRs_bayer_denoise, SSIMs_bayer_denoise = [], []
    PSNRs_bayer_noisy, SSIMs_bayer_noisy = [], []

    bar = tqdm(bm_loader)
    for input_bayer, gt_bayer, meta in bar:
        bar.set_description(meta.name)
        assert meta.bayer_pattern == 'BGGR'
        input_bayer, gt_bayer = RawUtils.bggr2rggb(input_bayer, gt_bayer)

        pred_bayer = denoiser.run(input_bayer, iso=meta.ISO)
        # pred_bayer = input_bayer  # dummy for test

        inp_rgb, pred_rgb, gt_rgb = RawUtils.bayer01_2_rgb01(
            input_bayer, pred_bayer, gt_bayer,
            wb_gain=meta.wb_gain, CCM=meta.CCM,
        )

        inp_rgb, pred_rgb, gt_rgb = RawUtils.bggr2rggb(inp_rgb, pred_rgb, gt_rgb)
        bar.set_description(meta.name+' ✓')
        assert cv2.imwrite("inp_rgb.png", (inp_rgb*255.0).astype(np.uint8))
        assert cv2.imwrite("pred_rgb.png", (pred_rgb*255.0).astype(np.uint8))
        assert cv2.imwrite("gt_rgb.png", (gt_rgb*255.0).astype(np.uint8))

        psnrs_rgb_denoise, ssims_rgb_denoise = [], []
        psnrs_rgb_noisy, ssims_rgb_noisy = [], []
        psnrs_bayer_denoise, ssims_bayer_denoise = [], []
        psnrs_bayer_noisy, ssims_bayer_noisy = [], []

        for x0, y0, x1, y1 in meta.ROIs:
            # ----- raw ----- #
            pred_patch_bayer = pred_bayer[y0:y1, x0:x1]
            gt_patch_bayer = gt_bayer[y0:y1, x0:x1]
            noisy_patch_bayer = input_bayer[y0:y1, x0:x1]

            psnr_bayer_denoise = skimage.metrics.peak_signal_noise_ratio(gt_patch_bayer, pred_patch_bayer)
            psnrs_bayer_denoise.append(float(psnr_bayer_denoise))
            psnr_bayer_noisy = skimage.metrics.peak_signal_noise_ratio(gt_patch_bayer, noisy_patch_bayer)
            psnrs_bayer_noisy.append(float(psnr_bayer_noisy))
            # ssim_bayer = skimage.metrics.structural_similarity(gt_patch_bayer, pred_patch_bayer, multichannel=True)
            # ssims_bayer.append(float(ssim))

            # ----- rgb ----- #
            pred_patch_rgb = pred_rgb[y0:y1, x0:x1]
            gt_patch_rgb = gt_rgb[y0:y1, x0:x1]
            noisy_patch_rgb = inp_rgb[y0:y1, x0:x1]

            psnr_rgb_denoise = skimage.metrics.peak_signal_noise_ratio(gt_patch_rgb, pred_patch_rgb)
            psnrs_rgb_denoise.append(float(psnr_rgb_denoise))
            psnr_rgb_noisy = skimage.metrics.peak_signal_noise_ratio(gt_patch_rgb, noisy_patch_rgb)
            psnrs_rgb_noisy.append(float(psnr_rgb_noisy))
            # ssim = skimage.metrics.structural_similarity(gt_patch, pred_patch, multichannel=True)
            # ssims.append(float(ssim))

        bar.set_description(meta.name+' ✓✓')
        print("ISO = ", meta.ISO)
        print("current psnrs_bayer (gt to denoise) = ", np.mean(psnrs_bayer_denoise))
        print("current psnrs_bayer (gt to noisy) = ", np.mean(psnrs_bayer_noisy))
        print("current psnrs_rgb (gt to denoise) = ", np.mean(psnrs_rgb_denoise)) 
        print("current psnrs_rgb (gt to noisy) = ", np.mean(psnrs_rgb_noisy)) 

        PSNRs_rgb_denoise = PSNRs_rgb_denoise + psnrs_rgb_denoise   # list append
        # SSIMs = SSIMs + ssims
        PSNRs_bayer_denoise = PSNRs_bayer_denoise + psnrs_bayer_denoise   # list append

    mean_psnr = np.mean(PSNRs_rgb_denoise)
    # mean_ssim = np.mean(SSIMs)
    print("mean PSNR:", mean_psnr)
    # print("mean SSIM:", mean_ssim)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=Path)
    parser.add_argument('--benchmark', type=Path)

    args = parser.parse_args()

    # args.benchmark = Path("D:/users/xiaoyaopan/PxyAI/DataSet/PMRID/PMRID/benchmark.json")
    # args.model = Path("D:/users/xiaoyaopan/PxyAI/PMRID_OFFICIAL/PMRID/models/torch_pretrained.ckp")

    bm_loader = BenchmarkLoader(args.benchmark.resolve())
    run_benchmark(args.model, bm_loader)

# vim: ts=4 sw=4 sts=4 expandtab
