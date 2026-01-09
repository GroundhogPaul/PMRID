import os 
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True, max_split_size_mb:128'
import torch
print(f"PYTORCH_CUDA_ALLOC_CONF: {os.environ.get('PYTORCH_CUDA_ALLOC_CONF', 'Not set')}")
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import torchvision.utils as vutils
from collections import defaultdict
import argparse
import cv2

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.RawDataset import create_dataloader, PMRIDRawDataset
from benchmark import BenchmarkLoader
from run_benchmark import Denoiser, KSigma, Official_Ksigma_params
from utilRaw import RawUtils

from models.net_torch import NetworkPMRID as Network

import os
import numpy as np
import glob
import time
from tqdm import tqdm
from utils.loss import calc_psnr

def train():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', 
                        default='models/PMRID_KSigma',
                        help='Location at which to save model logs and checkpoints.'
                        )
    parser.add_argument('--train_pattern', 
                        default='D:/image_database/SID/SID/Sony/long/*.ARW',
                        help='Pattern for directory containing source JPG images for training.'
                        )
    parser.add_argument('--test_pattern', 
                        default='D:/image_database/SID/SID/Sony/long_test/*.ARW',
                        help='Pattern for directory containing source JPG images for testing.'                   
                        )
    parser.add_argument('--image_size', type=int, default=1024)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--learning_rate', type=float, default=5e-4)
    parser.add_argument('--num_epochs', type=float, default=8000)
    parser.add_argument('--train_loss_log_step', type=int, default=500, help='Log train loss every N steps, step = batch') # 1000
    parser.add_argument('--eval_step', type=int, default=500, help='Log images to TensorBoard every N steps, step = batch') # 5000
    parser.add_argument('--resume', default=True, help='Whether to resume training')

    args = parser.parse_args()

    # visible_device_list代码端配置  2 3 1 0    <->    window任务管理器  GPU0 GPU1 GPU2 GPU3
    torch.cuda.empty_cache()
    device = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')
    model = Network().to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    criterion = torch.nn.L1Loss()

    # load checkpoint
    if args.resume:
        best_model_path, best_psnr = find_best_model(args.model_dir)
        if best_model_path:
            print(f"find best checkpoint: {best_model_path} (PSNR: {best_psnr:.2f})")

            checkpoint = torch.load(best_model_path, weights_only=False)
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_step = checkpoint['step']
            best_psnr = checkpoint['psnr']
            print(f"resume from step:{start_step}, best PSNR: {best_psnr:.2f}")
        else:
            start_step = 0
            print(f'not finding saved checkpoint, training a new model from step:0')  
    else:
        start_step = 0
        print(f'training a new model from step:0')


    dataset = PMRIDRawDataset(args.train_pattern, args.image_size, args.image_size, bPreLoadAll=False, device = device)
    train_loader = create_dataloader(dataset, args.batch_size)
    # test_loader = create_dataloader(args.test_pattern, args.image_size, args.image_size, args.batch_size)
    import pathlib as Path
    pathBenchMarkJson = Path.Path("D:/users/xiaoyaopan/PxyAI/DataSet/PMRID/PMRID/benchmark.json")
    bm_loader = BenchmarkLoader(pathBenchMarkJson.resolve())
    writer = SummaryWriter(os.path.join(args.model_dir, 'log'))
    
    # Track top 10 models by PSNR
    lst_top_models = defaultdict(list)
    lst_latest_models = defaultdict(list)
    os.makedirs(os.path.join(args.model_dir, 'top_models'), exist_ok=True)

    # ----- clear and create dump folder ----- #
    nSaveRemain = 20
    pathFolderNtrainImage = os.path.join(args.model_dir, 'First_N_train_image')
    pathFolderDump = os.path.join(args.model_dir, 'Dump')
    import shutil
    if not args.resume:
        shutil.rmtree(pathFolderNtrainImage, ignore_errors = True)
        shutil.rmtree(pathFolderDump, ignore_errors = True)
    os.makedirs(pathFolderNtrainImage, exist_ok=True)
    os.makedirs(pathFolderDump, exist_ok=True)

    nSaveTestCnt = 20
    step = start_step
    # ---------- start training ---------- #
    for epoch in range(args.num_epochs): 
        model.train()
        start_time = time.time()
        start_time_epoch = time.time()
        for batch_idx, (inputs_rggb_gt, inputs_rggb_noisy, inputs_rggb_noisy_k, meta_data) in enumerate(train_loader):
            
            optimizer.zero_grad()
            inp_scale = 256.0
            inputs_rggb_noisy_k = inputs_rggb_noisy_k * inp_scale # strange magic number from run_benchmark.py
            outputs_rggb_pred = model(inputs_rggb_noisy_k.to(torch.float32))
            outputs_rggb_pred = outputs_rggb_pred / inp_scale # strange magic number from run_benchmark.py

            kSigmaCur = KSigma(Official_Ksigma_params['K_coeff'], Official_Ksigma_params['B_coeff'], Official_Ksigma_params['anchor'])
            for ithImg, iso in enumerate(meta_data['iso']):
                outputs_rggb_pred[ithImg*4:ithImg*4+4] = kSigmaCur(outputs_rggb_pred[ithImg*4:ithImg*4+4] , iso, inverse=True)

            loss = criterion(outputs_rggb_pred, inputs_rggb_gt) / args.batch_size
            loss.backward()
            optimizer.step()

            if nSaveRemain > 0: # save the first nSaveRemain train image
                # ----- prepare test images for log ----- #
                noisy_bgr888 = dataset.ConvertDatasetImgToBGR888(inputs_rggb_noisy_k / inp_scale, meta_data, bKsigma=True, idx = 0)
                cv2.imwrite(os.path.join(pathFolderNtrainImage, f"{nSaveRemain}_Noisy.jpg"), noisy_bgr888)
                gt_bgr888 = dataset.ConvertDatasetImgToBGR888(inputs_rggb_gt, meta_data, idx = 0)
                cv2.imwrite(os.path.join(pathFolderNtrainImage, f"{nSaveRemain}_GT.jpg"), gt_bgr888)

                nSaveRemain -= 1
            
            step += 1
            if step % args.train_loss_log_step == 0:
                current_time = time.time() # s
                print(f'Epoch: {epoch}, Step: {step}, Batch: {batch_idx + 1}/{len(train_loader)}, Loss: {loss.item():.6f}, Time: {(current_time - start_time):.2f}s')
                start_time = current_time
                writer.add_scalar('train_loss', loss.item(), step)

            # evaluate
            if step % args.eval_step == 0:
            #     # Evaluation each epoch
            #     denoiser = Denoiser(model, kSigmaCur, device = device, inp_scale = inp_scale)
            #     # PSNRtest_psnr s_bayer_denoise = []
                test_loss = 0.0
                test_psnr = 0.0
            #     example_images = []  # Store example images for visualization

            #     bar = tqdm(bm_loader)
            #     with torch.no_grad():
            #         for input_bayer, gt_bayer, meta in bar:
            #             psnrs_bayer_denoise, ssims_bayer_denoise = [], []
            #             bar.set_description(meta.name)
            #             assert meta.bayer_pattern == 'BGGR'
            #             input_bayer_01, gt_bayer_01 = RawUtils.bggr2rggb(input_bayer, gt_bayer)
            #             gt_bayer_01 = torch.from_numpy(np.ascontiguousarray(gt_bayer_01)).cuda(device)
            #             input_bayer_01 = torch.from_numpy(np.ascontiguousarray(input_bayer_01)).cuda(device)
            #             pred_bayer_01 = denoiser.run(input_bayer_01, iso=meta.ISO)
            #             for x0, y0, x1, y1 in meta.ROIs:
            #                 # ----- raw ----- #
            #                 pred_patch_bayer_01 = pred_bayer_01[y0:y1, x0:x1]
            #                 gt_patch_bayer_01 = gt_bayer_01[y0:y1, x0:x1]

            #                 psnr_bayer_denoise = calc_psnr(gt_patch_bayer_01, pred_patch_bayer_01)
            #                 psnrs_bayer_denoise.append(float(psnr_bayer_denoise))

            #             # test_loss += criterion(gt_rggb_01, output_rggb_01).item()
            #             test_psnr += np.mean(psnrs_bayer_denoise)

            #             # # Store first batch of images for visualization
            #             # if len(example_images) == 0 and meta.ISO == 6400.0:
            #             #     labels_test = RawUtils.bayer01_2_rgb01(gt_bayer_01.cpu().numpy(), gamma=2.2, wb_gain=meta.wb_gain, CCM=meta.CCM)
            #             #     inputs_test = RawUtils.bayer01_2_rgb01(input_bayer_01.cpu().numpy(), gamma=2.2, wb_gain=meta.wb_gain, CCM=meta.CCM)
            #             #     outputs_test = RawUtils.bayer01_2_rgb01(pred_bayer_01.cpu().numpy(), gamma=2.2, wb_gain=meta.wb_gain, CCM=meta.CCM)
            #             #     labels_test = (labels_test*255.0).astype(np.uint8)
            #             #     inputs_test = (inputs_test*255.0).astype(np.uint8)
            #             #     outputs_test = (outputs_test*255.0).astype(np.uint8)
            #             #     cv2.imwrite("gt_rgb_from_benchmark.bmp", labels_test)
            #             #     cv2.imwrite("noisy_rgb_from_benchmark.bmp", inputs_test)
            #             #     cv2.imwrite("denoised_rgb_from_benchmark.bmp", outputs_test)
            #             #     pred_bayer_01
            #             #     example_images.append({
            #             #         'noisy_test': inputs_test,  # batch
            #             #         'denoised_test': outputs_test,
            #             #         'clean_test': labels_test
            #             #     })
            #             #     # break # test code
                # # log metrics
                # test_loss /= len(bm_loader)
                # test_psnr /= len(bm_loader)
                # print(f'Epoch: {epoch}, Test Loss: {test_loss:.6f}, Test PSNR: {test_psnr:.2f}')
                # writer.add_scalar('test_loss', test_loss, step)
                # writer.add_scalar('test_psnr', test_psnr, step)

                # log imagaes
                # images = example_images[0]
                # writer.add_image('Noisy_test', images['noisy_test'], epoch)
                # writer.add_image('Denoised_test', images['denoised_test'], epoch) 
                # writer.add_image('Clean_test', images['clean_test'], epoch)
                
                # Also log the difference between denoised and clean
                # diff = torch.abs(denoised_img_rgb - clean_img_rgb)
                # diff_grid = vutils.make_grid(diff[0].permute(2,0,1)) #, normalize=True)
                # writer.add_image('Difference', diff_grid, step)

                # Save and update models
                # torch.save(model.state_dict(), f'{args.model_dir}/model_{epoch}.pth')

                # ----- log train images ----- #
                train_psnr = calc_psnr(inputs_rggb_gt, outputs_rggb_pred)
                sDumpTrainPrefix = f"{epoch:04d}_{train_psnr:.2f}"

                sDumpTrainNoisy = os.path.join(pathFolderDump, sDumpTrainPrefix + "_Train_Noisy.bmp")
                noisy_bgr888 = dataset.ConvertDatasetImgToBGR888(inputs_rggb_noisy_k / inp_scale, meta_data, bKsigma=True, idx = 0)
                cv2.imwrite(sDumpTrainNoisy, noisy_bgr888)

                sDumpTrainPred = os.path.join(pathFolderDump, sDumpTrainPrefix + "_Train_Pred.bmp")
                pred_bgr888 = dataset.ConvertDatasetImgToBGR888(outputs_rggb_pred, meta_data, idx = 0)
                cv2.imwrite(sDumpTrainPred, pred_bgr888)

                sDumpTrainGT = os.path.join(pathFolderDump, sDumpTrainPrefix + "_Train_GT.bmp")
                gt_bgr888 = dataset.ConvertDatasetImgToBGR888(inputs_rggb_gt, meta_data, idx = 0)
                cv2.imwrite(sDumpTrainGT, gt_bgr888)

                save_checkpoint(lst_top_models, lst_latest_models, model, optimizer, epoch, test_psnr, args.model_dir)
        end_time_epoch = time.time()
        print(f'Epoch: {epoch}, Loss: {loss.item():.6f}, Time: {(end_time_epoch - start_time_epoch):.2f}s')

def save_checkpoint(lst_top_models, lst_lateset_models, model, optimizer, epoch, psnr, model_dir):
    os.makedirs(model_dir, exist_ok=True)
    nKeepTop = 10
    nKeepLatest = 100
    
    # ---------- top models ---------- #
    if len(lst_top_models) < nKeepTop or psnr > min(lst_top_models.keys()):
        # Remove the worst model if we already have 10
        if len(lst_top_models) >= nKeepTop:
            worst_psnr = min(lst_top_models.keys())
            os.remove(lst_top_models[worst_psnr][0])
            del lst_top_models[worst_psnr]        

        top_model_path = os.path.join(model_dir, 'top_models', f'top_model_psnr_{psnr:.2f}_epoch_{epoch}.pth')
        torch.save({
            'epoch':epoch,
            'state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'psnr':psnr}, 
            top_model_path)
        lst_top_models[psnr] = (top_model_path, epoch)

    # ---------- lateset models ---------- #
    lateset_model_path = os.path.join(model_dir, 'top_models', f'lateset_model_psnr_{psnr:.2f}_epoch_{epoch}.pth')
    torch.save({
        'epoch':epoch,
        'state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'psnr':psnr}, 
        lateset_model_path)
    lst_lateset_models[epoch] = lateset_model_path
    if len(lst_lateset_models) > nKeepLatest:
        oldest_epoch = min(lst_lateset_models.keys())
        os.remove(lst_lateset_models[oldest_epoch])
        del lst_lateset_models[oldest_epoch]        

def find_best_model(model_dir):
    if not os.path.exists(model_dir):
        return None, -1
    
    checkpoint_files = glob.glob(os.path.join(model_dir, 'top_models', 'top_model_psnr_*_step_*.pth'))
    if not checkpoint_files:
        return None, -1

    psnr_values = []
    for f in checkpoint_files:
        try:
            psnr = float(os.path.basename(f).split('_')[3])
            psnr_values.append(psnr)
        except:
            continue

    best_idx = np.argmax(psnr_values)
    return  checkpoint_files[best_idx], psnr_values[best_idx]

if __name__ == '__main__':
    train()