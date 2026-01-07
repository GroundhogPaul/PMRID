# add noise according to sigma to a vrf, to see if sigma is appropriate
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

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# ---------- read vrf ---------- #
# ----- read 'clean' vrf from Golden 4T inference result ----- #
sFolder = r"D:\users\xiaoyaopan\PxyAI\PMRID_OFFICIAL\PMRID\models\golden_4T"
assert os.path.exists(sFolder), f"Data folder does not exist: {sFolder}"
# idxVrf = 53
idxVrf = 33
vrf_files = glob.glob(os.path.join(sFolder, f"{idxVrf}_AI_Denoise.vrf"))
assert len(vrf_files) > 0, f"VRF file does not exist in folder: {os.path.join(sFolder, str(idxVrf))}"
assert len(vrf_files) == 1, f"Multiple VRF files found in folder: {os.path.join(sFolder, str(idxVrf))}"
sVrfPath = os.path.join(sFolder, vrf_files[0])

# sVrfCpyName = f"{idxVrf:02d}_noisy.vrf"
# sVrfCpyPath =  os.path.join(sOut_folder, sVrfCpyName)
# shutil.copy(sVrfPath, sVrfCpyPath)

# ----- read vrf info ----- #
vrfCur = vrf(sVrfPath)
ISO = vrfCur.m_ISO
SensorGain = vrfCur.m_nSensorGain
print(f"Using ISO: {ISO}, SensorGain: {SensorGain}")

# sVrfOutName = f"{idxVrf:02d}_{sImgSuffix}_denoise.vrf"
# sVrfOutPath =  os.path.join(sOut_folder, sVrfOutName)

black_level = vrfCur.m_BlackLevel
white_level = vrfCur.m_WhiteLevel 
blc01 = float(black_level) / white_level
dgain = 1.0

# ----- read vrf ----- #
clean_bayerGRBG = read_vrf(sVrfPath, vrfCur.m_W, vrfCur.m_H, black_level, dgain, white_level).astype(float)
clean_bayerGRBG = torch.from_numpy(np.ascontiguousarray(clean_bayerGRBG)).cuda(device)

# ---------- Add noise ---------- #
# ----- Get Sigma: CJ method ----- #
# sFileGain = "D:/image_database/jn1_mfnr_bestshot/gain.txt"
# sFileVar = "D:/image_database/jn1_mfnr_bestshot/var.txt"
# SensorGain = np.clip(SensorGain * 0.38, 1, 64)
# SensorGain = np.clip(SensorGain, 1, 64)
# var_shot, var_read = interpolate_gain_var(file_gain = sFileGain, file_var = sFileVar, TGain = SensorGain)

# ----- Get Sigma: LW method ----- #
var_shot, var_read = GetJin1ShotAndReadVar(SensorGain)
print("var_read = ", var_read, ", var_shot = ", var_shot)

# ---- add noise ----- #
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
var_GRBG = clean_bayerGRBG * var_shot + var_read
noise_bayerGRBG = torch.randn_like(clean_bayerGRBG) * torch.sqrt(var_GRBG)
print(noise_bayerGRBG.min(), noise_bayerGRBG.max())
noisy_bayerGRBG = clean_bayerGRBG + noise_bayerGRBG

# ----- save vrf ----- #
white_level_out = 2**10 - 1
black_level_out = 64
noisy_image = save_raw_image(noisy_bayerGRBG.detach().cpu().numpy(), "test.raw", white_level_out, black_level_out)
save_vrf_image(noisy_image, sVrfPath, f"{idxVrf}_AddNoise.vrf", white_level_out)

# ---------- Denoise ---------- #
# ----- read model ----- #
model_path =  "./models/TIM_BROOKS_AsMuchBlc_Discrete/top_models/lateset_model_psnr_0.00_epoch_7950.pth"
assert os.path.exists(model_path), f"Model file does not exist: {model_path}"

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
sOut_folder = os.path.join(sModel_folder, 'denoise_vrf_out_modify')
os.makedirs(sOut_folder, exist_ok=True)

noisy_bayerRGGB = torch.fliplr(noisy_bayerGRBG)
# ----- padd to 32 multiple ----- #
noisy_rggb = RawUtils.bayer2rggb(noisy_bayerRGGB)
noisy_rggb = noisy_rggb.permute(2, 0, 1).unsqueeze(0)  # [1,4,H,W]

B, C, H, W = noisy_rggb.shape
pad_h = (32 - H % 32) % 32
pad_w = (32 - W % 32) % 32
noisy_rggb = torch.nn.functional.pad(noisy_rggb, (0, pad_w, 0, pad_h), mode='constant', value = 0)

# ----- get variance map and concate ----- #
var_rggb = torch.sqrt(torch.clamp(noisy_rggb, 0, 1) * var_shot + var_read).to(torch.float32).to(device)
concat_rggb = torch.cat([noisy_rggb.to(torch.float32), var_rggb], dim=1)

# ----- forward ----- #
pred_rggb = net(concat_rggb)[0]  # [B,4,H,W]

# ----- depad ----- #
pred_rggb = pred_rggb[:, :H, :W]
pred_bayerRGGB = RawUtils.rggb2bayer(pred_rggb.permute(1, 2, 0)).detach().cpu().numpy()
pred_bayerGRBG = np.fliplr(pred_bayerRGGB)

# ----- save vrf ----- #
pred_bayerGRBG = np.clip(pred_bayerGRBG + blc01, 0.0, 1.0)
pred_bayerGRBG = save_raw_image(pred_bayerGRBG, "test.raw", white_level, 0)
save_vrf_image(pred_bayerGRBG, sVrfPath,  f"{idxVrf}_AddNoiseDenoise.vrf", white_level)