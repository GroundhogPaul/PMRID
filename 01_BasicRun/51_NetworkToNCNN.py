import utilBasicRun

import numpy as np
import skimage
import os
import re
import cv2
import pnnx
# from utils.KSigma import KSigma, Official_Ksigma_params 
# from utilVrf import vrf, read_vrf, save_vrf_image, save_raw_image, CFAPatternEnum, FlipBayerPattern2Pattern
# from models.net_torch import NetworkPMRID as Network
# from models.net_torch_SCH import Network_Level3_ch_off_bilinear as Network
# from models.net_torch_noahtcv import NOAHTCV as Network
from models.net_torch_noah_add import NOAH_L2_add as Network
import torch

if __name__ == '__main__':
    Wbayer, Hbayer = 1024, 1024
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # ---------- read model ---------- #
    # ----- assert ckpt paths ----- #
    model_folder =  "./runs/models/NOAH_L2_add/DumpCkpt"
    model_name = "008200_L0.0059.ckpt"
    model_path = os.path.join(model_folder, model_name)
    # assert os.path.exists(model_path), f"Model file does not exist: {model_path}"

    # ----- get model name -----
    path_parts = model_path.split('/')
    models_index = path_parts.index('models')
    model_name = path_parts[models_index + 1]
    print(model_name)

    # ----- output folder ----- #
    sModel_folder = os.path.dirname(os.path.dirname(model_path))

    # ----- load ckpt ----- #
    net = Network().to(device)
    # net.load_CKPT(str(model_path), device=torch.device(device))
    net.eval()

    dummy_input = torch.randn(1, 8, Hbayer//2, Wbayer//2).to(device)

    print(dummy_input.shape)
    print(dummy_input.dtype)

    opt_model = pnnx.export(net, model_name, dummy_input)