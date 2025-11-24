import utilBasicRun
from models.net_torch import *

model = NetworkPMRID()
sModelCheckPoint = r"D:\users\xiaoyaopan\PxyAI\PMRID_OFFICIAL\PMRID\models\torch_pretrained.ckp"

model.load_CKPT(sModelCheckPoint, torch.device('cpu'))

img = torch.randn(1, 4, 64, 64, device=torch.device('cpu'), dtype=torch.float32)
out = model(img)
print(out.size())