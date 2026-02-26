import utilBasicRun

import numpy as np
import skimage
import os
import re
import cv2
import pnnx
from utilRaw import RawUtils
from utils.KSigma import KSigma, Official_Ksigma_params 
from utilVrf import vrf, read_vrf, save_vrf_image, save_raw_image, CFAPatternEnum, FlipBayerPattern2Pattern
# from models.net_torch import NetworkPMRID as Network
from models.net_torch_SCH import Network_Level3_ch_off_bilinear as Network
import torch

if __name__ == '__main__':
    Wout, Hout = 4224, 3136
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # ---------- read model ---------- #
    # ----- assert ckpt paths ----- #
    model_folder, inp_scale =  "./runs/models/Huan2.4G_fiveK_subset230_default_color/top_models/", 256
    model_name = "latest_modelK_psnr0.00_e7803s218500_lr5.00e-04.pth"

    model_path = os.path.join(model_folder, model_name)
    assert os.path.exists(model_path), f"Model file does not exist: {model_path}"
    match = re.search(r'e(?P<epoch>\d+)s(?P<step>\d+)', model_name)
    epoch = int(match.group('epoch'))
    step = int(match.group('step'))

    # ----- get model name -----
    path_parts = model_path.split('/')
    models_index = path_parts.index('models')
    # sImgSuffix = path_parts[models_index + 1] + f"_netK{epoch}"
    sImgSuffix = "HuanKSigma"
    print("sImgSuffix = ", sImgSuffix)
    assert os.path.exists(model_path), f"Model file does not exist: {model_path}"

    # ----- output folder ----- #
    sModel_folder = os.path.dirname(os.path.dirname(model_path))

    # ----- load ckpt ----- #
    net = Network(mode="KSigma").to(device)
    net.load_CKPT(str(model_path), device=torch.device(device))
    net.eval()

    dummy_input = torch.randn(1, 4, Hout//2, Wout//2).to(device)

    print(dummy_input.shape)
    print(dummy_input.dtype)

    opt_model = pnnx.export(net, sImgSuffix, dummy_input)