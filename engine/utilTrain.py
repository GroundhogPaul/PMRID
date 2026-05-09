import sys, os
from pathlib import Path
# ----- add the parent folder to environment ----- 
ROOT = Path(__file__).resolve().parents[1]
ROOT = Path(__file__).parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

import utilBasic

import numpy as np
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
        self.device = config_yaml['device']

        self.folderDumpLog = os.path.join(self.output_dir, "DumpLog")
        os.makedirs(self.folderDumpLog, exist_ok=True)
        self.folderDump1stImg = os.path.join(self.output_dir, "Dump1stImg")
        os.makedirs(self.folderDump1stImg, exist_ok=True)
        self.folderDumpPred = os.path.join(self.output_dir, "DumpPred")
        os.makedirs(self.folderDumpPred, exist_ok=True)

        self.folderDumpCkpt = os.path.join(self.output_dir, "DumpCkpt")
        os.makedirs(self.folderDumpCkpt, exist_ok=True)

# def find_best_model(model_dir):
#     if not os.path.exists(model_dir):
#         return None, -1
    
#     checkpoint_files = glob.glob(os.path.join(model_dir, 'top_models', 'top_model_psnr_*_epoch_*.pth'))
#     if not checkpoint_files:
#         return None, -1

#     psnr_values = []
#     for f in checkpoint_files:
#         try:
#             psnr = float(os.path.basename(f).split('_')[3])
#             psnr_values.append(psnr)
#         except:
#             continue

#     best_idx = np.argmax(psnr_values)
#     return  checkpoint_files[best_idx], psnr_values[best_idx]


# # load checkpoint
# if args.resume:
#     best_model_path, best_psnr = find_best_model(args.model_dir)
#     if best_model_path:
#         print(f"find best checkpoint: {best_model_path} (PSNR: {best_psnr:.2f})")

#         checkpoint = torch.load(best_model_path, weights_only=False)
#         model.load_state_dict(checkpoint['state_dict'])
#         optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
#         start_epoch = checkpoint['epoch']
#         best_psnr = checkpoint['psnr']
#         print(f"resume from epoch:{start_epoch}, best PSNR: {best_psnr:.2f}")
#     else:
#         start_epoch = 0
#         print(f'not finding saved checkpoint, training a new model from epoch:0')  
#         args.resume = False
# else:
#     start_epoch = 0
#     print(f'training a new model from epoch:0')

# def save_checkpoint(lst_top_models, lst_lateset_models, model, optimizer, epoch, psnr, model_dir):
#     os.makedirs(model_dir, exist_ok=True)
#     nKeepTop = 10
#     nKeepLatest = 100
    
#     # ---------- top models ---------- #
#     if len(lst_top_models) < nKeepTop or psnr > min(lst_top_models.keys()):
#         # Remove the worst model if we already have 10
#         if len(lst_top_models) >= nKeepTop:
#             worst_psnr = min(lst_top_models.keys())
#             os.remove(lst_top_models[worst_psnr][0])
#             del lst_top_models[worst_psnr]        

#         top_model_path = os.path.join(model_dir, 'top_models', f'top_model_psnr_{psnr:.2f}_epoch_{epoch}.pth')
#         torch.save({
#             'epoch':epoch,
#             'state_dict': model.state_dict(),
#             'optimizer_state_dict': optimizer.state_dict(),
#             'psnr':psnr}, 
#             top_model_path)
#         lst_top_models[psnr] = (top_model_path, epoch)

#     # ---------- lateset models ---------- #
#     lateset_model_path = os.path.join(model_dir, 'top_models', f'lateset_model_psnr_{psnr:.2f}_epoch_{epoch}.pth')
#     torch.save({
#         'epoch':epoch,
#         'state_dict': model.state_dict(),
#         'optimizer_state_dict': optimizer.state_dict(),
#         'psnr':psnr}, 
#         lateset_model_path)
#     lst_lateset_models[epoch] = lateset_model_path
#     if len(lst_lateset_models) > nKeepLatest:
#         oldest_epoch = min(lst_lateset_models.keys())
#         os.remove(lst_lateset_models[oldest_epoch])
#         del lst_lateset_models[oldest_epoch]        