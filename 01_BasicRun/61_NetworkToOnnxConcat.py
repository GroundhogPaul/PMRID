import utilBasicRun
import torch

import onnx
import onnxruntime as ort
import os

# from models.net_torch_noahtcv import NOAHTCV_Level3 as network
from models.net_torch_noahtcv_group import NOAHTCVgroup_Level3 as network

onnx_path = 'onnx/NOAHgroup1x1_L3_512x512.onnx'
os.makedirs(os.path.dirname(onnx_path), exist_ok=True)
input_shape = (1, 4, 512, 512)

# 1. load pytorch model
model = network()
# pth_path = 'models/torch_pretrained.ckp'
# model.load_state_dict(torch.load(pth_path))
model.eval()

# 2. create dummy input
dummy_input = torch.randn(input_shape)
dummy_variance = torch.randn(input_shape)

# 3. 导出为ONNX
torch.onnx.export(
    model,
    (dummy_input, dummy_variance),
    onnx_path,
    export_params=True,       # 是否导出模型参数
    opset_version=11,         # ONNX算子集版本（推荐11+）
    do_constant_folding=True, # 是否优化常量折叠
    input_names=["noisy_img", "std_map"],    # 输入节点名称
    output_names=["denoised_img"],  # 输出节点名称
    # dynamic_axes={
    #     "input": {0: "batch_size"},  # 动态批次维度
    #     "output": {0: "batch_size"}
    # } if dynamic_batch else None
)
print(f"模型已成功导出至 {onnx_path}")