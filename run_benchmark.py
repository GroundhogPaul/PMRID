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
from utils import RawUtils
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


class Denoiser:

    def __init__(self, model_path: Path, ksigma: KSigma, inp_scale=256.0):
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


def run_benchmark(model_path, bm_loader: BenchmarkLoader):

    ksigma = KSigma(
        K_coeff=[0.0005995267, 0.00868861],
        B_coeff=[7.11772e-7, 6.514934e-4, 0.11492713],
        anchor=1600,
    )
    denoiser = Denoiser(model_path, ksigma)

    PSNRs, SSIMs = [], []

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

        psnrs = []
        ssims = []

        for x0, y0, x1, y1 in meta.ROIs:
            pred_patch = pred_rgb[y0:y1, x0:x1]
            gt_patch = gt_rgb[y0:y1, x0:x1]

            psnr = skimage.metrics.peak_signal_noise_ratio(gt_patch, pred_patch)
            # ssim = skimage.metrics.structural_similarity(gt_patch, pred_patch, multichannel=True)
            psnrs.append(float(psnr))
            # ssims.append(float(ssim))

        bar.set_description(meta.name+' ✓✓')
        print("current PSNR:", np.mean(PSNRs))

        PSNRs = PSNRs + psnrs   # list append
        # SSIMs = SSIMs + ssims

    mean_psnr = np.mean(PSNRs)
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
