import utilBasicRun

import numpy as np
import skimage
import os
import cv2
from utilRaw import RawUtils
from run_benchmark import KSigma, Official_Ksigma_params, Denoiser
from utilVrf import vrf, read_vrf, save_vrf_image, save_raw_image
from models.net_torch import NetworkPMRID as Network
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True, max_split_size_mb:128'
import torch
print(f"PYTORCH_CUDA_ALLOC_CONF: {os.environ.get('PYTORCH_CUDA_ALLOC_CONF', 'Not set')}")
torch.cuda.empty_cache()
import shutil
import glob

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
dgainPxy = 2

# ---------- read model ---------- #
# ----- assert ckpt paths ----- #
# model_path, inp_scale =  "./models/PMRID_pretrain/top_models/torch_pretrained.ckp", 256
# model_path, inp_scale =  "./models/PMRID_Jitter/top_models/top_model_psnr_50.57_step_505500.pth", 256
model_path, inp_scale =  "./models/PMRID_JitterBright1Contrast0/top_models/top_model_psnr_50.57_step_1437000.pth", 256
# model_path, inp_scale =  "./models/PMRID_JitterBright0Contrast1/top_models/top_model_psnr_50.46_step_322000.pth", 256
# model_path, inp_scale =  "./models/PMRID_test/top_models/top_model_psnr_49.36_step_166000.pth", 256

# ----- get model name -----
path_parts = model_path.split('/')
models_index = path_parts.index('models')
sImgSuffix = path_parts[models_index + 1]
print("sImgSuffix = ", sImgSuffix)
assert os.path.exists(model_path), f"Model file does not exist: {model_path}"

# ----- load ckpt ----- #
net = Network().to(device)
net.load_CKPT(str(model_path), device=torch.device(device))
net.eval()

# ----- output folder ----- #
sModel_folder = os.path.dirname(os.path.dirname(model_path))
sOut_folder = os.path.join(sModel_folder, 'denoise_vrf_out')
os.makedirs(sOut_folder, exist_ok=True)

# ---------- read vrf ---------- #
# ----- glob and copy input vrf ----- #
sFolder = r"D:\image_database\jn1_mfnr_bestshot\unpacked"
assert os.path.exists(sFolder), f"Data folder does not exist: {sFolder}"
for idxVrf in range(1, 64):
    vrf_files = glob.glob(os.path.join(sFolder, f"{idxVrf}/*.vrf"))
    assert len(vrf_files) > 0, f"VRF file does not exist in folder: {os.path.join(sFolder, str(idxVrf))}"
    assert len(vrf_files) == 1, f"Multiple VRF files found in folder: {os.path.join(sFolder, str(idxVrf))}"
    sVrfPath = os.path.join(sFolder, vrf_files[0])

    sVrfCpyName = f"{idxVrf:02d}_noisy.vrf"
    sVrfCpyPath =  os.path.join(sOut_folder, sVrfCpyName)
    shutil.copy(sVrfPath, sVrfCpyPath)

    # ----- read vrf info ----- #
    vrfCur = vrf(sVrfPath)
    ISO = vrfCur.m_ISO
    print(f"Using ISO: {ISO}")

    sVrfOutName = f"{idxVrf:02d}_denoise_{sImgSuffix}.vrf"
    sVrfOutPath =  os.path.join(sOut_folder, sVrfOutName)
    sVrfOutPathGain = os.path.splitext(sVrfOutPath)[0] + "_Gain" + os.path.splitext(sVrfOutName)[1] 

    black_level = 64
    white_level = 1023
    dgain = 1.0

    # ----- read vrf ----- #
    bayer01_GRBG_noisy = read_vrf(sVrfPath, vrfCur.m_W, vrfCur.m_H, black_level, dgain, white_level)
    bayer01_RGGB_noisy = np.fliplr(bayer01_GRBG_noisy)
    bayer01_RGGB_noisy = torch.from_numpy(np.ascontiguousarray(bayer01_RGGB_noisy)).cuda(device)

    # ---------- Denoise ---------- #
    kSigma = KSigma(
        K_coeff=Official_Ksigma_params["K_coeff"],
        B_coeff=Official_Ksigma_params["B_coeff"],
        anchor=Official_Ksigma_params["anchor"]
    )

    DenoiserCur = Denoiser(net, kSigma, device, inp_scale=inp_scale)

    bayer01_RGGB_denoise = DenoiserCur.run(bayer01_RGGB_noisy, iso=ISO)
    bayer01_RGGB_denoise = bayer01_RGGB_denoise.cpu().numpy() 
    bayer01_RGGB_denoise = np.clip(bayer01_RGGB_denoise, 0.0, 1.0)

    # ---------- save image ---------- #
    # ----- save png ----- #
    bayer01_BGGR_denoise = RawUtils.rggb2bggr(bayer01_RGGB_denoise)

    # TODO: use wb_gain from vrf
    # bgr01_denoise = RawUtils.bayer01_2_rgb01(bayer01_BGGR_denoise, wb_gain=[1.5156, 1.0, 1.7421], CCM=np.eye(3))
    bgr01_denoise = RawUtils.bayer01_2_rgb01(bayer01_BGGR_denoise, wb_gain=[vrfCur.m_AWB_Bgain, vrfCur.m_AWB_Ggain, vrfCur.m_AWB_Rgain], CCM=np.eye(3))
    bgr_denoise = (bgr01_denoise*255.0).astype(np.uint8)
    bgr_denoise = np.flipud(bgr_denoise)
    sBmpOutPath = sVrfOutPath.replace(".vrf", ".bmp")
    cv2.imwrite(sBmpOutPath, bgr_denoise)

    bgr_denoise_gainPxy = bgr_denoise.astype(float) * dgainPxy
    bgr_denoise_gainPxy = np.clip(bgr_denoise_gainPxy, 0, 255).astype(np.uint8)
    sBmpOutPathGain = sVrfOutPathGain.replace(".vrf", ".bmp")
    cv2.imwrite(sBmpOutPathGain, bgr_denoise_gainPxy)


    # ----- cal psnr to std ----- #
    # import skimage
    # psnr = skimage.metrics.peak_signal_noise_ratio(bgr_denoise, bgr_noisy)
    # print("psnr_bgr = ", psnr)
    # print(bayer01_RGGB_denoise.min(), bayer01_RGGB_denoise.max())
    # print(bayer01_RGGB_noisy.min(), bayer01_RGGB_noisy.max())
    # psnr = skimage.metrics.peak_signal_noise_ratio(bayer01_RGGB_denoise, bayer01_RGGB_noisy) 
    # print("psnr_bayer01 = ", psnr)

    # bgr_denoise_std = cv2.imread("denoise_rgb_std.bmp")
    # errNorm2 = np.linalg.norm(bgr_denoise.astype(np.float32) - bgr_denoise_std.astype(np.float32))
    # print("errNorm2 = ", errNorm2)

    # ----- save vrf ----- #
    out_ratio = 4  #out 12bit
    out_black_level = black_level * out_ratio  # 根据实际情况调整
    out_white_level = (white_level + 1) * out_ratio - 1
    bayer01_GRBG_denoise = np.fliplr(bayer01_RGGB_denoise)
    denoised_image = save_raw_image(bayer01_GRBG_denoise, sVrfOutPath.replace(".vrf", ".raw"), out_white_level, out_black_level)
    save_vrf_image(denoised_image, sVrfPath, sVrfOutPath, out_white_level)

    denoised_image_with_dgainPxy = denoised_image * dgainPxy
    save_vrf_image(denoised_image_with_dgainPxy, sVrfPath, sVrfOutPathGain, out_white_level)