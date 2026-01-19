import utilBasicRun

import numpy as np
import skimage
import os
import cv2
from utilRaw import RawUtils
from run_benchmark import Denoiser
from utils.KSigma import KSigma, Official_Ksigma_params 
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
model_path, inp_scale =  "./runs/models/PMRID_KSigmaLuoWen_1_64_16_Wholy/top_models/lateset_model_psnr_0.00_epoch_205.pth", 256
# model_path, inp_scale =  "./models/PMRID_JitterBright1Contrast0/top_models/top_model_psnr_50.57_step_1437000.pth", 256
# model_path, inp_scale =  "./models/PMRID_JitterBright0Contrast1/top_models/top_model_psnr_50.46_step_322000.pth", 256
# model_path, inp_scale =  "./models/PMRID_withBlc/top_models/top_model_psnr_49.41_step_212000.pth", 256
# model_path, inp_scale =  "./models/PMRID_KSigma/top_models/lateset_model_psnr_0.00_epoch_1549.pth", 256

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
# for idxVrf in range(33, 34):
for idxVrf in [33, 53]:
    vrf_files = glob.glob(os.path.join(sFolder, f"{idxVrf}/*.vrf"))
    assert len(vrf_files) > 0, f"VRF file does not exist in folder: {os.path.join(sFolder, str(idxVrf))}"
    assert len(vrf_files) == 1, f"Multiple VRF files found in folder: {os.path.join(sFolder, str(idxVrf))}"

    # ---------- case 1: denoise Jn1 ---------- #
    sVrfPath = os.path.join(sFolder, vrf_files[0])
    sVrfCpyName = f"{idxVrf:02d}_noisy.vrf"
    sVrfOutName = f"{idxVrf:02d}_{sImgSuffix}.vrf"

    # ---------- Case2: Denoise 'add noise to golden 4T output' ---------- #
    # sFolder = r"D:\users\xiaoyaopan\PxyAI\PMRID_OFFICIAL\PMRID"
    # assert os.path.exists(sFolder), f"Data folder does not exist: {sFolder}"
    # vrf_files = glob.glob(os.path.join(sFolder, f"{idxVrf:02d}_AddNoise.vrf"))
    # assert len(vrf_files) > 0, f"VRF file does not exist in folder: {os.path.join(sFolder, str(idxVrf))}"
    # assert len(vrf_files) == 1, f"Multiple VRF files found in folder: {os.path.join(sFolder, str(idxVrf))}"
    # sVrfPath = os.path.join(sFolder, vrf_files[0])

    # sVrfCpyName = f"{idxVrf:02d}_AddNoise.vrf"
    # sVrfOutName = f"{idxVrf:02d}_{sImgSuffix}_AddNoiseDenoise.vrf"

    # ----- read vrf info ----- #
    sVrfOutPath =  os.path.join(sOut_folder, sVrfOutName)
    sVrfCpyPath =  os.path.join(sOut_folder, sVrfCpyName)
    shutil.copy(sVrfPath, sVrfCpyPath)
    vrfCur = vrf(sVrfPath)
    ISO = vrfCur.m_ISO
    print(f"Using ISO: {ISO}")


    black_level = vrfCur.m_BlackLevel
    white_level = vrfCur.m_WhiteLevel
    dgain = 1.0

    # ----- read vrf ----- #
    bayer01_GRBG_noisy = read_vrf(sVrfPath, vrfCur.m_W, vrfCur.m_H, black_level, dgain, white_level, bClipBlc=True)
    bayer01_RGGB_noisy = np.fliplr(bayer01_GRBG_noisy)
    bayer01_RGGB_noisy = torch.from_numpy(np.ascontiguousarray(bayer01_RGGB_noisy)).cuda(device)

    # ---------- Denoise ---------- #
    kSigma = KSigma(
        K_coeff=Official_Ksigma_params["K_coeff"],
        B_coeff=Official_Ksigma_params["B_coeff"],
        anchor=Official_Ksigma_params["anchor"],
        # k = 0.00251 * 1023,
        # sigma = 1.265e-05 * 1023 * 1023, 
    )

    DenoiserCur = Denoiser(net, kSigma, device, inp_scale=inp_scale)

    bayer01_RGGB_denoise = DenoiserCur.run(bayer01_RGGB_noisy, iso=ISO)
    bayer01_RGGB_denoise = bayer01_RGGB_denoise.cpu().numpy() 
    bayer01_RGGB_denoise = np.clip(bayer01_RGGB_denoise, 0.0, 1.0)

    # ----- save vrf ----- #
    out_ratio = 4  #out 12bit
    out_black_level = black_level * out_ratio  # 根据实际情况调整
    out_white_level = (white_level + 1) * out_ratio - 1
    bayer01_GRBG_denoise = np.fliplr(bayer01_RGGB_denoise)
    denoised_image = save_raw_image(bayer01_GRBG_denoise, sVrfOutPath.replace(".vrf", ".raw"), out_white_level, out_black_level)
    save_vrf_image(denoised_image, sVrfPath, sVrfOutPath, out_white_level)