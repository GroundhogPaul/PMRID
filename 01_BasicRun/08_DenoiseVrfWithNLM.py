import utilBasicRun

import numpy as np
import skimage
import os
import re
import cv2
from utilRaw import RawUtils
from run_benchmark import DenoiserWithNLM
from utils.KSigma import KSigma, Official_Ksigma_params 
from utils.NLM import NLM_rggb_withPad
from utilVrf import vrf, read_vrf, save_vrf_image, save_raw_image, CFAPatternEnum, FlipBayerPattern2Pattern
from models.net_torch_SCH import Network_Level3_ch_off_bilinear_NLM as Network
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True, max_split_size_mb:128'
import torch
print(f"PYTORCH_CUDA_ALLOC_CONF: {os.environ.get('PYTORCH_CUDA_ALLOC_CONF', 'Not set')}")
torch.cuda.empty_cache()
import shutil
import glob

if __name__ == '__main__':
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # ---------- read model ---------- #
    # ----- assert ckpt paths ----- #
    model_folder, inp_scale =  "./runs/models/Huan2.4G_fiveK_withNLM_AddBackNoisy/top_models/", 256
    model_name = "latest_modelK_psnr0.00_e273s162900_lr5.00e-04.pth"

    model_path = os.path.join(model_folder, model_name)
    assert os.path.exists(model_path), f"Model file does not exist: {model_path}"
    match = re.search(r'e(?P<epoch>\d+)s(?P<step>\d+)', model_name)
    epoch = int(match.group('epoch'))
    step = int(match.group('step'))

    # ----- get model name -----
    path_parts = model_path.split('/')
    models_index = path_parts.index('models')
    sImgSuffix = path_parts[models_index + 1] + f"_netK{epoch}"
    print("sImgSuffix = ", sImgSuffix)
    assert os.path.exists(model_path), f"Model file does not exist: {model_path}"

    # ----- load ckpt ----- #
    net = Network(mode="KSigma").to(device)
    net.load_CKPT(str(model_path), device=torch.device(device))
    net.eval()

    # ----- output folder ----- #
    sModel_folder = os.path.dirname(os.path.dirname(model_path))
    sOut_folder = os.path.join(sModel_folder, 'denoise_vrf_out')
    os.makedirs(sOut_folder, exist_ok=True)

    # ---------- read vrf ---------- #
    # ----- glob and copy input vrf ----- #
    sFolder = r"D:\image_database\jn1_mfnr_bestshot\unpacked"
    # sFolder = r"D:\image_database\jn1_dark"
    assert os.path.exists(sFolder), f"Data folder does not exist: {sFolder}"
    # for idxVrf in range(1, 66):
    for idxVrf in [33, 53]:

        # ---------- case 1: denoise Jn1 ---------- #
        vrf_files = glob.glob(os.path.join(sFolder, f"{idxVrf}/*.vrf"))
        assert len(vrf_files) > 0, f"VRF file does not exist in folder: {os.path.join(sFolder, str(idxVrf))}"
        assert len(vrf_files) == 1, f"Multiple VRF files found in folder: {os.path.join(sFolder, str(idxVrf))}"
        sVrfPath = os.path.join(sFolder, vrf_files[0])
        sDngPath = sVrfPath.replace('.vrf', '.dng')
        sDngCpyName = f"{idxVrf:02d}_noisy.dng"
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
        sDngCpyPath =  os.path.join(sOut_folder, sDngCpyName)
        shutil.copy(sDngPath, sDngCpyPath)
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

        # ----- NLM ---- #
        bayer_GRBG_noisy = vrfCur.m_raw
        bayer_RGGB_noisy = FlipBayerPattern2Pattern(bayer_GRBG_noisy, vrfCur.m_CFAPatternNum, CFAPatternEnum.RGGB)
        bayer_RGGB_noisy = torch.from_numpy(np.ascontiguousarray(bayer_RGGB_noisy)).unsqueeze(0)
        bayer_RGGB_NLM = NLM_rggb_withPad(bayer_RGGB_noisy)

        bayer01_RGGB_NLM = torch.clamp(bayer_RGGB_NLM - black_level, 0, white_level).to(torch.float32)/white_level

        # ---------- Denoise ---------- #
        kSigma = KSigma(
            K_coeff=Official_Ksigma_params["K_coeff"],
            B_coeff=Official_Ksigma_params["B_coeff"],
            anchor=Official_Ksigma_params["anchor"],
            # k = 0.00251 * 1023,
            # sigma = 1.265e-05 * 1023 * 1023, 
        )

        DenoiserCur = DenoiserWithNLM(net, kSigma, device, inp_scale=inp_scale)

        bayer01_RGGB_denoise = DenoiserCur.run(bayer01_RGGB_noisy, bayer01_RGGB_NLM, iso=ISO)
        bayer01_RGGB_denoise = bayer01_RGGB_denoise.cpu().numpy() 
        bayer01_RGGB_denoise = np.clip(bayer01_RGGB_denoise, 0.0, 1.0)

        # ----- save vrf ----- #
        out_ratio = 4  #out 12bit
        out_black_level = black_level * out_ratio  # 根据实际情况调整
        out_white_level = (white_level + 1) * out_ratio - 1
        bayer01_GRBG_denoise = FlipBayerPattern2Pattern(bayer01_RGGB_denoise, CFAPatternEnum.RGGB, vrfCur.m_CFAPatternNum)
        denoised_image = save_raw_image(bayer01_GRBG_denoise, sVrfOutPath.replace(".vrf", ".raw"), out_white_level, out_black_level, bSave = False)
        save_vrf_image(denoised_image, sVrfPath, sVrfOutPath, out_white_level)