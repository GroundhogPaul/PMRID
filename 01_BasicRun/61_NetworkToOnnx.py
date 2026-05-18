import utilBasicRun
import torch

import onnx
import onnxruntime as ort

from models.net_torch import Network

pth_path = 'models/torch_pretrained.ckp'
onnx_path = 'onnx/torch_pretrainded.onnx'
input_shape = (1, 4, 512, 512)

# 1. load pytorch model
model = Network()
model.load_state_dict(torch.load(pth_path))
model.eval()

# 2. create dummy input
dummy_input = torch.randn(input_shape)

# 3. 导出为ONNX
torch.onnx.export(
    model,
    dummy_input,
    onnx_path,
    export_params=True,       # 是否导出模型参数
    opset_version=11,         # ONNX算子集版本（推荐11+）
    do_constant_folding=True, # 是否优化常量折叠
    input_names=["input"],    # 输入节点名称
    output_names=["output"],  # 输出节点名称
    # dynamic_axes={
    #     "input": {0: "batch_size"},  # 动态批次维度
    #     "output": {0: "batch_size"}
    # } if dynamic_batch else None
)
print(f"模型已成功导出至 {onnx_path}")
