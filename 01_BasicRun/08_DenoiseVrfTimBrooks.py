import utilBasicRun

import numpy as np
import skimage
import os
import cv2
import re
from RawContainer.utilRaw import RawUtils
from RawContainer.utilVrf import vrf, read_vrf, save_vrf_image, save_raw_image, CFAPatternEnum, FlipBayerPattern2Pattern
from models.net_torch_noahtcv_group import NOAHTCVgroup_Level3 as Network
from engine.utilTrain import TrainState, DenoiserConcat, DenoiserVrf
import torch
import shutil
import glob

if __name__ == '__main__':
    # ---------- read model ---------- #
    # ----- assert ckpt paths ----- #
    # model_path, model_name =  "runs/models/NOAHgroupL3/DumpCkpt/023400_L0.0089.ckpt", "NOAHgroupL3"
    model_path, model_name =  "runs/models/NOAHgroupL3vrf/DumpCkpt/041400_L0.0016.ckpt", "NOAHgroupL3"
    assert os.path.exists(model_path), f"Model file does not exist: {model_path}"

    myState = TrainState(sModelName=model_name, model=Network, tpm=None)
    myState.LoadModelFromPath(model_path, device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'))

    # ----- output folder ----- #
    sModel_folder = os.path.dirname(os.path.dirname(model_path))
    sOut_folder = os.path.join(sModel_folder, 'denoise_vrf_out')
    os.makedirs(sOut_folder, exist_ok=True)

    # ---------- read vrf ---------- #
    # ----- case 1: 1~64 img ----- #
    sFolder = r"D:\image_database\jn1_mfnr_bestshot\unpacked"
    assert os.path.exists(sFolder), f"Data folder does not exist: {sFolder}"
    idxVrf = 33
    vrf_files = glob.glob(os.path.join(sFolder, f"{idxVrf}/*.vrf"))
    assert len(vrf_files) > 0, f"VRF file does not exist in folder: {os.path.join(sFolder, str(idxVrf))}"
    assert len(vrf_files) == 1, f"Multiple VRF files found in folder: {os.path.join(sFolder, str(idxVrf))}"
    sVrfPath = os.path.join(sFolder, vrf_files[0])

    sVrfCpyName = f"{idxVrf:02d}_noisy.vrf"
    sVrfCpyPath =  os.path.join(sOut_folder, sVrfCpyName)
    
    sVrfOutName = f"{idxVrf:02d}_{myState.sModelName}_{myState.batch_idx_total}.vrf"

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
    # sFolder, sFileName = r"D:\users\xiaoyaopan\PxyAI\DataSet\Jn1\s5kjn1_calibration_raw\blc_unpack", r"gain_64_unpack.vrf"
    # assert os.path.exists(sFolder), f"Data folder does not exist: {sFolder}"
    # sVrfPath = os.path.join(sFolder, sFileName)
    # assert os.path.exists(sVrfPath), f"Data file does not exist: {sVrfPath}"

    # sVrfCpyName = os.path.splitext(os.path.basename(sVrfPath))[0] + "_noise.vrf"
    # sVrfOutName = os.path.splitext(os.path.basename(sVrfPath))[0] + "_" + sImgSuffix + "_denoise.vrf"

    # # # ----- copy input vrf ----- #
    sVrfCpyPath =  os.path.join(sOut_folder, sVrfCpyName)
    shutil.copy(sVrfPath, sVrfCpyPath)

    # # -----denoise and save ----- #
    sVrfOutPath =  os.path.join(sOut_folder, sVrfOutName)
    DenoiserVrf(sVrfPath, sVrfOutPath, 
                sFolderNoiseParam = r"D:\users\xiaoyaopan\PxyAI\DataSet\SensorParam\jn1\NoiseParam", 
                model = myState.model, 
                mode = "Concat")

    print("sVrfOutPath = ", sVrfOutPath)