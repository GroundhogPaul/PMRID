import sys, os
from pathlib import Path
# ----- add the parent folder to environment ----- 
ROOT = Path(__file__).resolve().parents[1]
ROOT = Path(__file__).parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

import utilBasic
from RawContainer.utilVrf import vrf, read_vrf, save_vrf_image, save_raw_image, CFAPatternEnum, FlipBayerPattern2Pattern
from RawContainer.utilRaw import RawUtils
from SensorNoise.UtilSensorNoise import interpolate_gain_var_folder

import numpy as np
import torch
import yaml

class TrainParam:
    def __init__(self, sPathTrainParamYml):
        assert os.path.exists(sPathTrainParamYml), f"train param yml file not found: {sPathTrainParamYml}"
        with open(sPathTrainParamYml, 'r') as f:
            config_yaml = yaml.safe_load(f)

        self.output_dir = config_yaml['output_dir']
        sFolderTrainParamYml = os.path.dirname(sPathTrainParamYml)
        assert Path(self.output_dir) == Path(sFolderTrainParamYml), "Yes, I insist TrainParam.yml in the output folder"

        self.train_pattern = config_yaml['train_pattern']
        self.test_pattern = config_yaml['test_pattern']

        self.image_size = int(config_yaml['image_size'])
        self.batch_size = int(config_yaml['batch_size'])
        self.lr = float(config_yaml['lr'])
        self.img_per_step = int(config_yaml['img_per_step'])
        assert self.img_per_step > self.batch_size
        assert self.img_per_step % self.batch_size == 0, f"img_per_step={self.img_per_step}, batch_size={self.batch_size}"
        self.batch_per_step = self.img_per_step // self.batch_size
        self.num_step = int(config_yaml['num_step'])

        self.cal_train_loss_per_step = int(config_yaml['cal_train_loss_per_step']) # means write train loss and train img
        self.cal_eval_loss_per_step = int(config_yaml['cal_eval_loss_per_step']) # means dump ckpt and eval img
        self.resume = bool(config_yaml['resume'])
        self.resume_Mode = config_yaml['resume_mode']
        self.device = config_yaml['device']

        self.folderDumpLog = os.path.join(self.output_dir, "DumpLog")
        os.makedirs(self.folderDumpLog, exist_ok=True)
        self.folderDump1stImg = os.path.join(self.output_dir, "Dump1stImg")
        os.makedirs(self.folderDump1stImg, exist_ok=True)
        self.folderDumpPred = os.path.join(self.output_dir, "DumpPred")
        os.makedirs(self.folderDumpPred, exist_ok=True)

        self.folderDumpCkpt = os.path.join(self.output_dir, "DumpCkpt")
        os.makedirs(self.folderDumpCkpt, exist_ok=True)

class TrainState:
    def __init__(self, sModelName, model, tpm:TrainParam):
        assert isinstance(sModelName, str)
        self.eval = True
        if tpm is not None:
            assert isinstance(tpm, TrainParam)
            self.eval = False 
        else:
            print(" !!! input train param is None, eval mode !!!")

        self.tpm = tpm
        self.sModelName = sModelName
        self.model = model()
        self.model.to(tpm.device)
        self.optimizer = None

        self.lr = -1 # current lr
        self.step = 0
        self.batch_idx_total = 0
        self.train_loss = torch.tensor(-1)
        self.eval_loss = torch.tensor(-1)

        from torch.utils.tensorboard import SummaryWriter
        if self.eval == False: # only write train tqdm for train mode, not eval mode
            self.writer = SummaryWriter(self.tpm.folderDumpLog)
    
    def printStatus(self, interval=10):
        if self.batch_idx_total % interval == 0:
            print(f"batch_idx_total={self.batch_idx_total}, step={self.step}, loss={self.train_loss.item():.4f}")
    
    def LogStatus(self):
        self.writer.add_scalar('train_loss', self.train_loss.item(), self.step)
        self.writer.add_scalar('lr', self.lr, self.step)
    
    def SaveModel(self, folderDumpCkpt):
        print("  dump  ")
        # eval_train = self.eval_loss.item()
        sModelDump = os.path.join(folderDumpCkpt, f"{self.step:06d}_L{self.train_loss.item():.4f}.ckpt")
        torch.save({
            'sModelName': self.sModelName,
            'state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),

            'lr': self.lr,
            'step': self.step,
            'batch_idx_total': self.batch_idx_total,
            'train_loss':self.train_loss,
            'eval_loss':self.eval_loss
            }, sModelDump)
        if self.eval == False: # only write train tqdm for train mode, not eval mode
            self.LogStatus()
    
    def LoadModel(self):
        if not self.tpm.resume:
            print("resume mode off, training a new model from step:0")
            return

        assert self.tpm.resume_Mode == "Latest" or self.tpm.resume_Mode == "BestTrain" or self.tpm.resume_Mode == "BestEval"
        assert os.path.exists(self.tpm.folderDumpCkpt)
        path_ckpt = self._SelectBestModel(self.tpm.resume_Mode)
        if path_ckpt is None:
            print(f"!!! no checkpoint found in {self.tpm.folderDumpCkpt}, training a new model from step:0")
            return path_ckpt

        self.LoadModelFromPath(path_ckpt, self.tpm.device)
        return path_ckpt
    
    def LoadModelFromPath(self, path_ckpt, device):
        assert os.path.exists(path_ckpt), f"Model file does not exist: {path_ckpt}"
        ckpt = torch.load(path_ckpt, weights_only=False)
        self.model.load_CKPT(sCKPT = path_ckpt, device=device)
        if self.eval:
            self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])

        self.lr = ckpt['lr']
        self.step = ckpt['step']
        self.batch_idx_total = ckpt['batch_idx_total']
        self.train_loss = ckpt['train_loss']
        self.eval_loss = ckpt['eval_loss']

        print(f"load model from {path_ckpt}, step:{self.step}, batch: {self.batch_idx_total}")
        print("current model name: ", self.sModelName, ", loaded model name: ", ckpt['sModelName'])
    
    def _SelectBestModel(self, sMode:str):
        assert sMode == "Latest" or sMode == "BestTrain" or sMode == "BestEval"
        import glob
        checkpoint_files = glob.glob(os.path.join(self.tpm.folderDumpCkpt, '*.ckpt'))
        if not checkpoint_files:
            return None
        
        scoreBest = float('inf')
        best_ckpt = None
        for f in checkpoint_files:
            ckpt = torch.load(f, weights_only=False)
            if sMode == "Latest":
                score = -ckpt['step'] # the smaller step, the newer the model
            elif sMode == "BestTrain":
                score = ckpt['train_loss'].item()
            elif sMode == "BestEval":
                score = ckpt['eval_loss'].item()
            
            if score < scoreBest:
                scoreBest = score
                best_ckpt = f

        return best_ckpt

    def nextBatch(self):
        self.batch_idx_total += 1

# ---------- Denoise ---------- #
def DenoiserConcat(noisy_bayerRGGB, net, SensorGain, sFolderNoiseParam):
    assert isinstance(noisy_bayerRGGB, torch.Tensor), f"Input must be a pytorch tensor, current type: {type(noisy_bayerRGGB)}"

    # ----- padd to 32 multiple ----- #
    noisy_rggb = RawUtils.bayer2rggb(noisy_bayerRGGB)
    noisy_rggb = noisy_rggb.permute(2, 0, 1).unsqueeze(0)  # [1,4,H,W]

    B, C, H, W = noisy_rggb.shape
    pad_h = (32 - H % 32) % 32
    pad_w = (32 - W % 32) % 32
    noisy_rggb = torch.nn.functional.pad(noisy_rggb, (0, pad_w, 0, pad_h), mode='constant', value = 0)

    # ----- Get Sigma: LW method ----- #
    varRead, stdShot = interpolate_gain_var_folder(sFolderNoiseParam, SensorGain)

    # ----- cal var and concat ----- #
    print("stdShot = ", stdShot, ", varRead = ", varRead)
    noisy_rggb = torch.clamp(noisy_rggb, 0, 1)
    var_rggb = torch.sqrt(noisy_rggb * stdShot + varRead).to(torch.float32)

    # --------- forward --------- #
    pred_rggb = net(noisy_rggb, var_rggb)[0].detach()  # [B,4,H,W]
    pred_rggb = torch.clamp(pred_rggb, 0, 1)
    # print(torch.min(noisy_rggb), torch.max(noisy_rggb), torch.mean(noisy_rggb))
    # print(torch.min(pred_rggb), torch.max(pred_rggb), torch.mean(pred_rggb))

    # ----- depad ----- #
    pred_rggb = pred_rggb[:, :H, :W]
    pred_bayerRGGB = RawUtils.rggb2bayer(pred_rggb.permute(1, 2, 0)).detach().cpu().numpy()

    return pred_bayerRGGB

def DenoiserVrf(sVrfPath, sVrfOutPath, sFolderNoiseParam, model, mode):
    assert mode == "Concat" or mode == "KSigma", f"Unsupported mode: {mode}. Supported modes are 'Concat' and 'KSigma'."
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
    noisy_bayerGRBG = read_vrf(sVrfPath, vrfCur.m_W, vrfCur.m_H, black_level, dgain, white_level, bClipBlc=True)
    noisy_bayerRGGB = np.fliplr(noisy_bayerGRBG)
    device = next(model.parameters()).device
    noisy_bayerRGGB = torch.from_numpy(np.ascontiguousarray(noisy_bayerRGGB)).to(device)
    if mode == "Concat":
        pred_bayerRGGB = DenoiserConcat(noisy_bayerRGGB, model, SensorGain, sFolderNoiseParam)
    if mode == "KSigma":
        pred_bayerRGGB = DenoiserKSigma(noisy_bayerRGGB, model, SensorGain, sFolderNoiseParam)

    # # ----- save vrf ----- #
    out_ratio = 4  #out 12bit
    out_black_level = black_level * out_ratio  # 根据实际情况调整
    out_white_level = (white_level + 1) * out_ratio - 1
    pred_bayerGRBG = np.fliplr(pred_bayerRGGB)
    # bayer01_GRBG_denoise = np.clip(bayer01_GRBG_denoise, 0, 1)
    denoised_image = save_raw_image(pred_bayerGRBG, sVrfOutPath.replace(".vrf", ".raw"), out_white_level, out_black_level)
    save_vrf_image(denoised_image, sVrfPath, sVrfOutPath, out_white_level)

    return
