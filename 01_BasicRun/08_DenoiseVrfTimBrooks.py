import utilBasicRun

import numpy as np
import skimage
import os
import cv2
from utilRaw import RawUtils
from run_benchmark import KSigma, Official_Ksigma_params
from utilVrf import vrf, read_vrf, save_vrf_image, save_raw_image 
from models.net_torch import NetworkTimBrooks as Network
import torch
import shutil
import glob
from PlotAgainShotRead import interpolate_gain_var, GetJin1ShotAndReadVar


# ---------- Denoise ---------- #
def Denoiser(noisy_bayerRGGB, device, SensorGain):
    if hasattr(noisy_bayerRGGB, 'permute'):    # Pytorch
        noisy_bayerRGGB = noisy_bayerRGGB.cuda(device)
    elif hasattr(noisy_bayerRGGB, 'shape'):      # Numpy
        noisy_bayerRGGB = torch.from_numpy(np.ascontiguousarray(noisy_bayerRGGB)).cuda(device)
    else:
        raise NotImplementedError("Input must be a numpy array or pytorch tensor, current type: {}".format(type(noisy_bayerRGGB)))

    # ----- padd to 32 multiple ----- #
    noisy_rggb = RawUtils.bayer2rggb(noisy_bayerRGGB)
    noisy_rggb = noisy_rggb.permute(2, 0, 1).unsqueeze(0)  # [1,4,H,W]

    B, C, H, W = noisy_rggb.shape
    pad_h = (32 - H % 32) % 32
    pad_w = (32 - W % 32) % 32
    noisy_rggb = torch.nn.functional.pad(noisy_rggb, (0, pad_w, 0, pad_h), mode='constant', value = 0)

    # ----- Get Sigma: LW method ----- #
    varShot, varRead = GetJin1ShotAndReadVar(SensorGain)

    # ratio = 1023.0/(1023.0-64.0)
    # varShot = varShot * ratio * ratio
    # varRead = varRead * ratio

    # ----- cal var and concat ----- #
    print("varShot = ", varShot, ", varRead = ", varRead)
    var_rggb = torch.sqrt(torch.clamp(noisy_rggb, 0, 1) * varShot + varRead).to(torch.float32).to(device)
    concat_rggb = torch.cat([noisy_rggb.to(torch.float32), var_rggb], dim=1)

    # --------- forward --------- #
    pred_rggb = net(concat_rggb)[0]  # [B,4,H,W]

    # ----- depad ----- #
    pred_rggb = pred_rggb[:, :H, :W]
    pred_bayerRGGB = RawUtils.rggb2bayer(pred_rggb.permute(1, 2, 0)).detach().cpu().numpy()

    return pred_bayerRGGB

def DenoiserVrf(sVrfPath, sVrfOutPath):
    # ----- read vrf info ----- #
    vrfCur = vrf(sVrfPath)
    ISO = vrfCur.m_ISO
    SensorGain = vrfCur.m_nSensorGain
    print(f"Using ISO: {ISO}, SensorGain: {SensorGain}")

    black_level = vrfCur.m_BlackLevel
    white_level = vrfCur.m_WhiteLevel 
    blc01 = float(black_level) / white_level
    dgain = 1.0

    # ----- read vrf ----- #
    # bayer01_GRBG_noisy = read_vrf(sVrfPath, vrfCur.m_W, vrfCur.m_H, black_level, dgain, white_level)
    noisy_bayerGRBG = read_vrf(sVrfPath, vrfCur.m_W, vrfCur.m_H, black_level, dgain, white_level, bClipBlc=False)
    noisy_bayerRGGB = np.fliplr(noisy_bayerGRBG)
    # noisy_bayerRGGB = noisy_bayerGRBG
    noisy_bayerRGGB = torch.from_numpy(np.ascontiguousarray(noisy_bayerRGGB)).cuda(device)

    pred_bayerRGGB = Denoiser(noisy_bayerRGGB, device, SensorGain)

    # ----- save vrf ----- #
    out_ratio = 4  #out 12bit
    out_black_level = black_level * out_ratio  # 根据实际情况调整
    out_white_level = (white_level + 1) * out_ratio - 1
    pred_bayerGRBG = np.fliplr(pred_bayerRGGB)
    # bayer01_GRBG_denoise = np.clip(bayer01_GRBG_denoise, 0, 1)
    denoised_image = save_raw_image(pred_bayerGRBG, sVrfOutPath.replace(".vrf", ".raw"), out_white_level, out_black_level)
    save_vrf_image(denoised_image, sVrfPath, sVrfOutPath, out_white_level)

    return

if __name__ == '__main__':
    # ---------- read model ---------- #
    # ----- assert ckpt paths ----- #
    # model_path =  "./models/TIM_BROOKS_test_AsMuchJin1Blc_AWB/top_models/lateset_model_psnr_0.00_epoch_7950.pth"
    model_path =  "./models/TIM_BROOKS_AsMuchBlc_SingleNoise064/top_models/lateset_model_psnr_0.00_epoch_5450.pth"
    # model_path =  "./models/TIM_BROOKS_AsMuchBlc_Discrete/top_models/top_model_psnr_0.00_epoch_7950.pth"
    assert os.path.exists(model_path), f"Model file does not exist: {model_path}"

    # ----- get model name -----
    path_parts = model_path.split('/')
    models_index = path_parts.index('models')
    sImgSuffix = path_parts[models_index + 1]
    print("sImgSuffix = ", sImgSuffix)
    assert os.path.exists(model_path), f"Model file does not exist: {model_path}"

    # ----- load ckpt ----- #
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    net = Network().to(device)
    net.load_CKPT(str(model_path), device=torch.device(device))
    net.eval()

    # ----- output folder ----- #
    sModel_folder = os.path.dirname(os.path.dirname(model_path))
    sOut_folder = os.path.join(sModel_folder, 'denoise_vrf_out')
    os.makedirs(sOut_folder, exist_ok=True)

    # ---------- read vrf ---------- #
    # ----- case 1: 1~64 img ----- #
    # sFolder = r"D:\image_database\jn1_mfnr_bestshot\unpacked"
    # assert os.path.exists(sFolder), f"Data folder does not exist: {sFolder}"
    # idxVrf = 33
    # vrf_files = glob.glob(os.path.join(sFolder, f"{idxVrf}/*.vrf"))
    # assert len(vrf_files) > 0, f"VRF file does not exist in folder: {os.path.join(sFolder, str(idxVrf))}"
    # assert len(vrf_files) == 1, f"Multiple VRF files found in folder: {os.path.join(sFolder, str(idxVrf))}"
    # sVrfPath = os.path.join(sFolder, vrf_files[0])

    # sVrfCpyName = f"{idxVrf:02d}_noisy.vrf"
    # sVrfCpyPath =  os.path.join(sOut_folder, sVrfCpyName)
    
    # sVrfOutName = f"{idxVrf:02d}_{sImgSuffix}_denoise.vrf"

    # ---------- Case2: Denoise 'add noise to golden 4T output' ---------- #
    # sFolder = r"D:\users\xiaoyaopan\PxyAI\PMRID_OFFICIAL\PMRID"
    # assert os.path.exists(sFolder), f"Data folder does not exist: {sFolder}"
    # idxVrf = 53
    # vrf_files = glob.glob(os.path.join(sFolder, f"{idxVrf:02d}_AddNoise.vrf"))
    # assert len(vrf_files) > 0, f"VRF file does not exist in folder: {os.path.join(sFolder, str(idxVrf))}"
    # assert len(vrf_files) == 1, f"Multiple VRF files found in folder: {os.path.join(sFolder, str(idxVrf))}"
    # sVrfPath = os.path.join(sFolder, vrf_files[0])

    # sVrfCpyName = f"{idxVrf:02d}_AddNoise.vrf"
    # sVrfOutName = f"{idxVrf:02d}_{sImgSuffix}_AddNoiseDenoise.vrf"

    # ----- case 3: calibration img ----- #
    # sFolder, sFileName = r"D:\users\xiaoyaopan\PxyAI\DataSet\Jn1\s5kjn1_noise_calibration_raw", r"optical_black/64x_unpack.vrf"
    # sFolder, sFileName = r"D:\users\xiaoyaopan\PxyAI\DataSet\Jn1\s5kjn1_noise_calibration_raw", r"noise_ccm/ccm_64x_1.vrf"
    sFolder, sFileName = r"D:\users\xiaoyaopan\PxyAI\DataSet\Jn1\s5kjn1_calibration_raw\blc_unpack", r"gain_64_unpack.vrf"
    assert os.path.exists(sFolder), f"Data folder does not exist: {sFolder}"
    sVrfPath = os.path.join(sFolder, sFileName)
    assert os.path.exists(sVrfPath), f"Data file does not exist: {sVrfPath}"

    sVrfCpyName = os.path.splitext(os.path.basename(sVrfPath))[0] + "_noise.vrf"
    sVrfOutName = os.path.splitext(os.path.basename(sVrfPath))[0] + "_" + sImgSuffix + "_denoise.vrf"

    # # ----- copy input vrf ----- #
    sVrfCpyPath =  os.path.join(sOut_folder, sVrfCpyName)
    shutil.copy(sVrfPath, sVrfCpyPath)

    # -----denoise and save ----- #
    sVrfOutPath =  os.path.join(sOut_folder, sVrfOutName)
    DenoiserVrf(sVrfPath, sVrfOutPath)

    print("sVrfOutPath = ", sVrfOutPath)