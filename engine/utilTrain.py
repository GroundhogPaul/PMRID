import sys, os
from pathlib import Path
# ----- add the parent folder to environment ----- 
ROOT = Path(__file__).resolve().parents[1]
ROOT = Path(__file__).parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

import utilBasic

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
    def __init__(self, sModelName, tpm:TrainParam):
        assert isinstance(sModelName, str)
        assert isinstance(tpm, TrainParam)
        self.tpm = tpm
        self.sModelName = sModelName
        self.model = None
        self.optimizer = None

        self.lr = -1 # current lr
        self.step = 0
        self.batch_idx_total = 0
        self.train_loss = torch.tensor(-1)
        self.eval_loss = torch.tensor(-1)

        from torch.utils.tensorboard import SummaryWriter
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

        ckpt = torch.load(path_ckpt, weights_only=False)
        self.model.load_state_dict(ckpt['state_dict'])
        self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])

        self.lr = ckpt['lr']
        self.step = ckpt['step']
        self.batch_idx_total = ckpt['batch_idx_total']
        self.train_loss = ckpt['train_loss']
        self.eval_loss = ckpt['eval_loss']

        print(f"resume from step:{self.step}, batch: {self.batch_idx_total}")
    
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