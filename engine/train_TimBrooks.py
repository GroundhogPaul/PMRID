import os
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

import sys, os
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)
import utilTrain
from utilTrain import ROOT, TrainParam, TrainState

import torch
import torch.optim as optim
import torchvision.utils as vutils
from collections import defaultdict
import argparse
import cv2

from data.RawDataset import DumpBgr888
from data.RawDatasetTimBrooks import RawDatasetTimBrooks, create_dataloader
from benchmark import BenchmarkLoader
# from run_benchmark import Denoiser, KSigma, Official_Ksigma_params

import time
import numpy as np
# import glob
# import time
# from tqdm import tqdm
# from utils.loss import calc_psnr
from learn_rate import lr_triangle

seed = 37
import random
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False   # 关闭自动调优

def train(sModelName, sPathTrainParamYml, Network):
    tpm = TrainParam(sPathTrainParamYml) # TrainParaM

    myState = TrainState(sModelName, model = Network, tpm = tpm)
    myState.optimizer = optim.Adam(myState.model.parameters(), lr=tpm.lr, weight_decay=1e-5)
    criterion = torch.nn.L1Loss()

    if "jpg" in tpm.train_pattern[0]:
        from ImgDataset import CropDatasetJpg as CropDataset
    elif "vrf" in tpm.train_pattern[0]:
        from ImgDataset import CropDatasetVrf as CropDataset
    else:
        assert False
    dataset = RawDatasetTimBrooks(
        tpm.train_pattern, tpm.image_size, tpm.image_size, device=tpm.device, CropDataset=CropDataset)
    train_loader = create_dataloader(dataset, tpm.batch_size)

    # ----- Log and Dump stuff ----- #
    print("batch_per_step = ", tpm.batch_per_step)
    nDump1st = 20 

    myState.LoadModel()

    # ---------- start training ---------- #
    myState.model.train()
    while True:
        for batch_idx, (BChHW_gt, BChHW_noisy, BChHW_var, meta_datas) in enumerate(train_loader):
            # ----- exit mechanism ----- #
            myState.step = myState.batch_idx_total // tpm.batch_per_step
            if myState.step >= tpm.num_step:
                break

            # ----- adjust lr ---- #
            # myState.lr = lr_triangle(myState.step, epochMax = tpm.num_step, lrMax = tpm.lr)
            myState.lr = tpm.lr
            for param_group in myState.optimizer.param_groups:
                param_group['lr'] = myState.lr

            # ----- train ----- #
            myState.optimizer.zero_grad()
            BChHW_pred = myState.model(BChHW_noisy, BChHW_var)
            myState.train_loss = criterion(BChHW_pred, BChHW_gt)
            myState.train_loss.backward()
            myState.optimizer.step()

            myState.printStatus()

            # ----- log ----- #
            if myState.batch_idx_total % (tpm.batch_per_step*tpm.cal_train_loss_per_step) == 0:
                myState.SaveModel(tpm.folderDumpCkpt)
                DumpBgr888(dataset, BChHW_noisy[0], meta_datas[0], myState.step, "Noisy0", tpm.folderDumpPred)
                DumpBgr888(dataset, BChHW_gt[0], meta_datas[0], myState.step, "GT0", tpm.folderDumpPred)
                DumpBgr888(dataset, BChHW_pred[0], meta_datas[0], myState.step, "Pred0", tpm.folderDumpPred)
                DumpBgr888(dataset, BChHW_var[0], meta_datas[0], myState.step, "NoiseStd", tpm.folderDumpPred)

            # ----- dump several image ----- #
            if nDump1st > 0: # save the first nSaveRemain train image
                DumpBgr888(dataset, BChHW_noisy[0], meta_datas[0], batch_idx, "Noisy0", tpm.folderDump1stImg)
                DumpBgr888(dataset, BChHW_gt[0], meta_datas[0], batch_idx, "GT0", tpm.folderDump1stImg)
                DumpBgr888(dataset, BChHW_pred[0], meta_datas[0], batch_idx, "Pred0", tpm.folderDump1stImg)
                DumpBgr888(dataset, BChHW_var[0], meta_datas[0], myState.step, "NoiseStd", tpm.folderDumpPred)
                nDump1st -= 1

            myState.nextBatch()

        if myState.step >= tpm.num_step:
            break

if __name__ == '__main__':
    sTrainFolder = "D:/users/xiaoyaopan/PxyAI/PMRID_OFFICIAL/PMRID/runs/models"

    sModelName, sTrainParamYml = "NOAHgroupL3jpg", "TrainArg.yaml"
    sModelFolder, sPathTrainParamYml = os.path.join(sTrainFolder, sModelName), os.path.join(sTrainFolder, sModelName, sTrainParamYml)
    if str(sModelFolder) not in sys.path:
        sys.path.append(str(sModelFolder))
    from net_torch_noahtcv_group import NOAHTCVgroup_Level3 as Network
    
    train(sModelName, sPathTrainParamYml, Network)

    # # ----- using BenchMark as test load ----- #
    # # test_loader = create_dataloader(args.test_pattern, args.image_size, args.image_size, args.batch_size)
    # import pathlib as Path
    # pathBenchMarkJson = Path.Path("D:/users/xiaoyaopan/PxyAI/DataSet/PMRID/PMRID/benchmark.json")
    # bm_loader = BenchmarkLoader(pathBenchMarkJson.resolve())
    
    # # Track top 10 models by PSNR
    # lst_top_models = defaultdict(list)
    # lst_latest_models = defaultdict(list)
    # os.makedirs(os.path.join(args.model_dir, 'top_models'), exist_ok=True)

    # # ----- clear and create dump folder ----- #
    # nSaveRemain = 20
    # pathFolderNtrainImage = os.path.join(args.model_dir, 'First_N_train_image')
    # pathFolderDump = os.path.join(args.model_dir, 'Dump')
    # import shutil
    # if not args.resume:
    #     shutil.rmtree(pathFolderNtrainImage, ignore_errors = True)
    #     shutil.rmtree(pathFolderDump, ignore_errors = True)
    # os.makedirs(pathFolderNtrainImage, exist_ok=True)
    # os.makedirs(pathFolderDump, exist_ok=True)

    # nSaveTestCnt = 1
    # # ---------- start training ---------- #
    # for epoch in range(args.num_epochs): 
    #     model.train()
    #     start_time = time.time()
    #     for batch_idx, (inputs_rggb_gt, inputs_rggb_noisy, inputs_rggb_variance, meta_data) in enumerate(train_loader):

    #         inputs_rggb_concat = torch.cat([inputs_rggb_noisy, inputs_rggb_variance], dim=1)  # concat noisy image and variance map
    #         optimizer.zero_grad()
    #         outputs_rggb_pred = model(inputs_rggb_concat.to(torch.float32))

    #         train_loss = criterion(outputs_rggb_pred, inputs_rggb_gt)
    #         train_loss.backward()
    #         optimizer.step()

    #         if nSaveRemain > 0: # save the first nSaveRemain train image
    #             # ----- prepare test images for log ----- #
    #             noisy_bgr888 = dataset.ConvertDatasetImgToBGR888(inputs_rggb_noisy, meta_data, 0)
    #             cv2.imwrite(os.path.join(pathFolderNtrainImage, f"{nSaveRemain}_Noisy.jpg"), noisy_bgr888)
    #             gt_bgr888 = dataset.ConvertDatasetImgToBGR888(inputs_rggb_gt, meta_data, 0)
    #             cv2.imwrite(os.path.join(pathFolderNtrainImage, f"{nSaveRemain}_GT.jpg"), gt_bgr888)

    #             nSaveRemain -= 1

    #     current_time = time.time() # s
    #     print(f'Epoch: {epoch}, Batch: {batch_idx + 1}/{len(train_loader)}, TrainLoss: {train_loss.item():.6f}, Time: {(current_time - start_time):.2f}s')
    #     if epoch % args.train_loss_log_epoch == 0:
    #         writer.add_scalar('train_loss', train_loss.item(), epoch)
            
    #     # evaluate
    #     if epoch % args.eval_epoch == 0:
    #         # ----- evaluate and save evaluate immage ----- #
    #         test_loss = 0.0
    #         test_psnr = 0 
    #         example_images = []  # Store example images for visualization

    #         # bar = tqdm(bm_loader)

    #         # iSaveTestCnt = 0
    #         # with torch.no_grad():
    #             # for input_bayer, gt_bayer, meta in bar:
    #             #     psnrs_bayer_denoise, ssims_bayer_denoise = [], []
    #             #     bar.set_description(meta.name)
    #             #     assert meta.bayer_pattern == 'BGGR'
    #             #     input_bayer_01, gt_bayer_01 = RawUtils.bggr2rggb(input_bayer, gt_bayer)
    #             #     input_bayer_01 = torch.from_numpy(np.ascontiguousarray(input_bayer_01)).to(device)
    #             #     gt_bayer_01 = torch.from_numpy(np.ascontiguousarray(gt_bayer_01)).to(device)

    #             #     # ----- bayer to rggb ----- #
    #             #     input_rggb_01 = RawUtils.bayer2rggb(input_bayer_01) 
    #             #     input_rggb_01 = input_rggb_01.unsqueeze(0)  # [1, H, W, 4]
    #             #     input_rggb_01 = input_rggb_01.permute(0, 3, 1, 2).to(device)

    #             #     # ----- padd to 32 multiple ----- #
    #             #     B, C, H, W = input_rggb_01.shape
    #             #     pad_h = (32 - H % 32) % 32
    #             #     pad_w = (32 - W % 32) % 32
    #             #     input_rggb_01 = torch.nn.functional.pad(input_rggb_01, (0, pad_w, 0, pad_h), mode='constant', value = 0)

    #             #     # ----- concat variance map ----- #
    #             #     k, sigma = KSigmaCur.GetKSigma(iso=meta.ISO)
    #             #     k = k / 1024
    #             #     sigma = sigma / 1024 / 1024
    #             #     # print("\n  test set: k:", k, " sigma:", sigma)
    #             #     variance_map = torch.sqrt(input_rggb_01 * k + sigma).to(torch.float32).to(device)
    #             #     input_rggb_01_concat = torch.cat([input_rggb_01.to(torch.float32), variance_map], dim=1)

    #             #     # ----- inference ----- #
    #             #     pred_rggb_01 = model(input_rggb_01_concat)[0]  # [B,4,H,W]

    #             #     # ----- depad ----- #
    #             #     pred_rggb_01 = pred_rggb_01[:, :H, :W]
    #             #     pred_bayer_01 = RawUtils.rggb2bayer(pred_rggb_01.permute(1, 2, 0))

    #             #     # ----- calc psnr ----- #
    #             #     for x0, y0, x1, y1 in meta.ROIs:
    #             #         # ----- raw ----- #
    #             #         pred_patch_bayer_01 = pred_bayer_01[y0:y1, x0:x1]
    #             #         gt_patch_bayer_01 = gt_bayer_01[y0:y1, x0:x1]

    #             #         psnr_bayer_denoise = calc_psnr(gt_patch_bayer_01, pred_patch_bayer_01)
    #             #         psnrs_bayer_denoise.append(float(psnr_bayer_denoise))

    #             #     test_loss += criterion(gt_patch_bayer_01, pred_patch_bayer_01).item()
    #             #     test_psnr += np.mean(psnrs_bayer_denoise)

    #                 # Store the test image
    #                 # if len(example_images) == 0 and meta.ISO == 6400.0:
    #                     # iSaveTestCnt = (iSaveTestCnt + 1) % 8
    #                     # if iSaveTestCnt != nSaveTestCnt:
    #                     #     continue
    #                     # nSaveTestCnt = (nSaveTestCnt + 1) % 8
    #                     # noisy_test = RawUtils.bayer01_2_rgb01(input_bayer_01.cpu().numpy(), gamma=2.2, wb_gain=meta.wb_gain, CCM=np.array(meta.CCM))
    #                     # pred_test = RawUtils.bayer01_2_rgb01(pred_bayer_01.cpu().numpy(), gamma=2.2, wb_gain=meta.wb_gain, CCM=np.array(meta.CCM))
    #                     # noisy_test = (noisy_test*255.0).astype(np.uint8)
    #                     # pred_test = (pred_test*255.0).astype(np.uint8)
    #                     # example_images.append({'noisy_test': noisy_test,
    #                     #                        'pred_test': pred_test,
    #                     #                     #    'gt_test': gt_test
    #                     #                     })
    #                     # break
    #         # ---------- log ---------- #
    #         # ----- log metrics ----- #
    #         # test_loss /= len(bm_loader)
    #         # test_psnr /= len(bm_loader)
    #         test_loss = 0
    #         test_psnr = 0
    #         print(f'Epoch: {epoch}, Test Loss: {test_loss:.6f}, Test PSNR: {test_psnr:.2f}')
    #         writer.add_scalar('test_loss', test_loss, epoch)
    #         writer.add_scalar('test_psnr', test_psnr, epoch)

    #         # ----- log test images ----- #
    #         print("iSaveTestCnt = ", iSaveTestCnt, ", nSaveTestCnt = ", nSaveTestCnt)
    #         if len(example_images) > 0:
    #             images = example_images[0]
    #             sDumpTestPrefix = f"{epoch:04d}_{test_psnr:.2f}"
    #             sDumpTestNoisy = os.path.join(pathFolderDump, sDumpTestPrefix + "_Test_Noisy.bmp")
    #             cv2.imwrite(sDumpTestNoisy, images['noisy_test'])
    #             sDumpTestPred = os.path.join(pathFolderDump, sDumpTestPrefix + "_Test_Pred.bmp")
    #             cv2.imwrite(sDumpTestPred, images['pred_test'])

    #         # ----- log train images ----- #
    #         train_psnr = calc_psnr(inputs_rggb_gt, outputs_rggb_pred)
    #         sDumpTrainPrefix = f"{epoch:04d}_{train_psnr:.2f}"

    #         sDumpTrainNoisy = os.path.join(pathFolderDump, sDumpTrainPrefix + "_Train_Noisy.bmp")
    #         noisy_bgr888 = dataset.ConvertDatasetImgToBGR888(inputs_rggb_noisy, meta_data, 0)
    #         cv2.imwrite(sDumpTrainNoisy, noisy_bgr888)

    #         sDumpTrainPred = os.path.join(pathFolderDump, sDumpTrainPrefix + "_Train_Pred.bmp")
    #         pred_bgr888 = dataset.ConvertDatasetImgToBGR888(outputs_rggb_pred, meta_data, 0)
    #         cv2.imwrite(sDumpTrainPred, pred_bgr888)

    #         sDumpTrainGT = os.path.join(pathFolderDump, sDumpTrainPrefix + "_Train_GT.bmp")
    #         gt_bgr888 = dataset.ConvertDatasetImgToBGR888(inputs_rggb_gt, meta_data, 0)
    #         cv2.imwrite(sDumpTrainGT, gt_bgr888)
                
    #         save_checkpoint(lst_top_models, lst_latest_models, model, optimizer, epoch, test_psnr, args.model_dir)
                
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
