import utilBasicRun

import numpy as np
import skimage
import os
import re
import cv2
from utilRaw import RawUtils
from run_benchmark import Denoiser
from utils.KSigma import KSigma, Official_Ksigma_params 
from utilVrf import vrf, read_vrf, save_vrf_image, save_raw_image, CFAPatternEnum, FlipBayerPattern2Pattern

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True, max_split_size_mb:128'
import torch
print(f"PYTORCH_CUDA_ALLOC_CONF: {os.environ.get('PYTORCH_CUDA_ALLOC_CONF', 'Not set')}")
torch.cuda.empty_cache()
import ncnn

def test_inference(model_param_path, model_bin_path, BChHW01_RGGB_noisy_k):
    assert os.path.exists(model_param_path), f"Model file does not exist: {model_param_path}"
    assert os.path.exists(model_bin_path), f"Model file does not exist: {model_bin_path}"

    print(BChHW01_RGGB_noisy_k.shape)
    # torch.manual_seed(0)
    # in0 = torch.rand(1, 4, 1560, 2104, dtype=torch.float)
    out = []

    with ncnn.Net() as net:
        net.load_param(model_param_path)
        net.load_model(model_bin_path)

        with net.create_extractor() as ex:
            BChHW01_RGGB_noisy_k = BChHW01_RGGB_noisy_k.squeeze(0).numpy()
            BChHW01_RGGB_noisy_k = np.ascontiguousarray(BChHW01_RGGB_noisy_k)
            ex.input("in0", ncnn.Mat(BChHW01_RGGB_noisy_k).clone())

            _, out0 = ex.extract("out0")
            out = np.array(out0)
            out = np.ascontiguousarray(out)
            out = torch.from_numpy(out)
            out = out.unsqueeze(0)

    return out
    # if len(out) == 1:
    #     return out[0]
    # else:
    #     return tuple(out)

if __name__ == "__main__":
    device = 'cpu'
    # ---------- read model ---------- #
    # ----- assert ckpt paths ----- #
    model_folder, inp_scale =  "./runs/models/Huan2.4G_fiveK_subset230_default_color/NCNN/", 256
    model_name = "latest_modelK_psnr0.00_e7803s218500_lr5.00e-04.pth"

    model_param_path = os.path.join(model_folder, "Huan_KSigma_4224x3136.ncnn.param")
    model_bin_path = os.path.join(model_folder, "Huan_KSigma_4224x3136.ncnn.bin")

    # ---------- read vrf ---------- #
    # ----- glob and copy input vrf ----- #
    sVrfPath = r"D:/image_database/jn1_mfnr_bestshot/unpacked/33/5_unpacked.vrf"
    assert os.path.exists(sVrfPath), f"Data folder does not exist: {sVrfPath}"

    vrfCur = vrf(sVrfPath)
    ISO = vrfCur.m_ISO
    print(f"Using ISO: {ISO}")

    black_level = vrfCur.m_BlackLevel
    white_level = vrfCur.m_WhiteLevel
    dgain = 1.0

    # ----- read vrf ----- #
    bayer01_GRBG_noisy = np.clip((vrfCur.m_raw - black_level), 0, white_level).astype(np.single)/white_level
    bayer01_RGGB_noisy = FlipBayerPattern2Pattern(bayer01_GRBG_noisy, vrfCur.m_CFAPatternNum, CFAPatternEnum.RGGB)
    bayer01_RGGB_noisy = torch.from_numpy(np.ascontiguousarray(bayer01_RGGB_noisy)).to(device)

    # ---------- Denoise ---------- #
    kSigma = KSigma(
        K_coeff=Official_Ksigma_params["K_coeff"],
        B_coeff=Official_Ksigma_params["B_coeff"],
        anchor=Official_Ksigma_params["anchor"],
        # k = 0.00251 * 1023,
        # sigma = 1.265e-05 * 1023 * 1023, 
    )

    DenoiserCur = Denoiser(None, kSigma, device, inp_scale=inp_scale)
    BChHW01_RGGB_noisy = DenoiserCur.pre_process(bayer01_RGGB_noisy, padH = 3136//2, padW = 4224//2)
    BChHW01_RGGB_noisy_k = kSigma(BChHW01_RGGB_noisy, ISO) * inp_scale

    BChHW01_RGGB_pred_k = test_inference(model_param_path, model_bin_path, BChHW01_RGGB_noisy_k)
    BChHW01_RGGB_pred_k = BChHW01_RGGB_pred_k / inp_scale
    BChHW01_RGGB_pred_k = BChHW01_RGGB_pred_k.detach()
    BChHW01_RGGB_pred = kSigma(BChHW01_RGGB_pred_k, ISO, inverse = True)
    bayer01_RGGB_denoise = DenoiserCur.post_process(BChHW01_RGGB_pred)

    bayer01_RGGB_denoise = bayer01_RGGB_denoise.cpu().numpy() 
    bayer01_RGGB_denoise = np.clip(bayer01_RGGB_denoise, 0.0, 1.0)

    # ----- save vrf ----- #
    sVrfOutPath = "./ncnn.vrf"
    out_ratio = 4  #out 12bit
    out_black_level = black_level * out_ratio  # 根据实际情况调整
    out_white_level = (white_level + 1) * out_ratio - 1
    bayer01_GRBG_denoise = FlipBayerPattern2Pattern(bayer01_RGGB_denoise, CFAPatternEnum.RGGB, vrfCur.m_CFAPatternNum)
    denoised_image = save_raw_image(bayer01_GRBG_denoise, sVrfOutPath.replace(".vrf", ".raw"), out_white_level, out_black_level, bSave = False)
    save_vrf_image(denoised_image, sVrfPath, sVrfOutPath, out_white_level)