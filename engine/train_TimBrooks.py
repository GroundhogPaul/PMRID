import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import torchvision.utils as vutils
from collections import defaultdict
import argparse
import cv2

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.RawDataset import create_dataloader, TimBrooksRawDataset, RawArrayToTensor 
#from dataset_SID import create_dataloader
from benchmark import BenchmarkLoader
from run_benchmark import Denoiser, KSigma, Official_Ksigma_params
from utilRaw import RawUtils

from models.net_torch import NetworkTimBrooks as Network

import os
import numpy as np
import glob
import time
from tqdm import tqdm
from utils.loss import calc_psnr

def train():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', 
                        default='models/TIM_BROOKS_test',
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
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--num_epochs', type=float, default=8000)
    parser.add_argument('--train_loss_log_step', type=int, default=100, help='Log train loss every N steps') # 1000
    parser.add_argument('--eval_step', type=int, default=500, help='Log images to TensorBoard every N steps') # 5000
    parser.add_argument('--resume', default=True, help='Whether to resume training')

    args = parser.parse_args()

    # visible_device_list代码端配置  2 3 1 0    <->    window任务管理器  GPU0 GPU1 GPU2 GPU3
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


    dataset = TimBrooksRawDataset(args.train_pattern, args.image_size, args.image_size)
    train_loader = create_dataloader(dataset, args.batch_size)
    # test_loader = create_dataloader(args.test_pattern, args.image_size, args.image_size, args.batch_size)
    import pathlib as Path
    pathBenchMarkJson = Path.Path("D:/users/xiaoyaopan/PxyAI/DataSet/PMRID/PMRID/benchmark.json")

    bm_loader = BenchmarkLoader(pathBenchMarkJson.resolve())
    KSigmaCur = KSigma(Official_Ksigma_params['K_coeff'], Official_Ksigma_params['B_coeff'], Official_Ksigma_params['anchor'])

    writer = SummaryWriter(os.path.join(args.model_dir, 'log'))
    
    # Track top 10 models by PSNR
    top_models = defaultdict(list)
    os.makedirs(os.path.join(args.model_dir, 'top_models'), exist_ok=True)

    for epoch in range(args.num_epochs): 
        model.train()
        start_time = time.time()
        for batch_idx, (inputs_rggb, inputs_rggb_noisy, inputs_rggb_variance, meta_data) in enumerate(train_loader):
            inputs_rggb_noisy = inputs_rggb_noisy.permute(0, 3, 1, 2).to(device)
            inputs_rggb = inputs_rggb.permute(0, 3, 1, 2).to(device)
            inputs_rggb_variance = inputs_rggb_variance.permute(0, 3, 1, 2).to(device)

            inputs_rggb_concat = torch.cat([inputs_rggb_noisy, inputs_rggb_variance], dim=1)  # concat noisy image and variance map
            optimizer.zero_grad()
            outputs = model(inputs_rggb_concat.to(torch.float32))

            loss = criterion(outputs, inputs_rggb)
            loss.backward()
            optimizer.step()
            
            step = start_step + batch_idx + 1 + epoch * len(train_loader)
            if step % args.train_loss_log_step == 0:
                current_time = time.time() # s
                print(f'Epoch: {epoch}, Step: {step}, Batch: {batch_idx + 1}/{len(train_loader)}, Loss: {loss.item():.6f}, Time: {(current_time - start_time):.2f}s')
                start_time = current_time
                writer.add_scalar('train_loss', loss.item(), step)

            # evaluate
            if step % args.eval_step == 0:
                test_loss = 0.0
                test_psnr = 0 
                example_images = []  # Store example images for visualization

                bar = tqdm(bm_loader)
                with torch.no_grad():
                    for input_bayer, gt_bayer, meta in bar:
                        psnrs_bayer_denoise, ssims_bayer_denoise = [], []
                        bar.set_description(meta.name)
                        assert meta.bayer_pattern == 'BGGR'
                        input_bayer_01, gt_bayer_01 = RawUtils.bggr2rggb(input_bayer, gt_bayer)
                        input_bayer_01 = torch.from_numpy(np.ascontiguousarray(input_bayer_01)).to(device)
                        gt_bayer_01 = torch.from_numpy(np.ascontiguousarray(gt_bayer_01)).to(device)

                        # ----- bayer to rggb ----- #
                        input_rggb_01 = RawUtils.bayer2rggb(input_bayer_01) 
                        input_rggb_01 = input_rggb_01.unsqueeze(0)  # [1, H, W, 4]
                        input_rggb_01 = input_rggb_01.permute(0, 3, 1, 2).to(device)

                        # ----- padd to 32 multiple ----- #
                        B, C, H, W = input_rggb_01.shape
                        pad_h = (32 - H % 32) % 32
                        pad_w = (32 - W % 32) % 32
                        input_rggb_01 = torch.nn.functional.pad(input_rggb_01, (0, pad_w, 0, pad_h), mode='constant', value = 0)

                        # ----- concat variance map ----- #
                        k, sigma = KSigmaCur.GetKSigma(iso=meta.ISO)
                        k = k / 1024
                        sigma = sigma / 1024 / 1024
                        # print("\n  test set: k:", k, " sigma:", sigma)
                        variance_map = torch.sqrt(input_rggb_01 * k + sigma).to(torch.float32).to(device)
                        input_rggb_01_concat = torch.cat([input_rggb_01.to(torch.float32), variance_map], dim=1)

                        # ----- inference ----- #
                        pred_rggb_01 = model(input_rggb_01_concat)[0]  # [B,4,H,W]

                        # ----- depad ----- #
                        pred_rggb_01 = pred_rggb_01[:, :H, :W]
                        pred_bayer_01 = RawUtils.rggb2bayer(pred_rggb_01.permute(1, 2, 0))

                        # ----- calc psnr ----- #
                        for x0, y0, x1, y1 in meta.ROIs:
                            # ----- raw ----- #
                            pred_patch_bayer_01 = pred_bayer_01[y0:y1, x0:x1]
                            gt_patch_bayer_01 = gt_bayer_01[y0:y1, x0:x1]

                            psnr_bayer_denoise = calc_psnr(gt_patch_bayer_01, pred_patch_bayer_01)
                            psnrs_bayer_denoise.append(float(psnr_bayer_denoise))

                        # test_loss += criterion(gt_rggb_01, output_rggb_01).item()
                        test_psnr += np.mean(psnrs_bayer_denoise)

                        # Store first batch of images for visualization
                        if len(example_images) == 0 and meta.ISO == 6400.0:
                            labels_test = RawUtils.bayer01_2_rgb01(gt_bayer_01.cpu().numpy(), gamma=2.2, wb_gain=meta.wb_gain, CCM=meta.CCM)
                            inputs_test = RawUtils.bayer01_2_rgb01(input_bayer_01.cpu().numpy(), gamma=2.2, wb_gain=meta.wb_gain, CCM=meta.CCM)
                            outputs_test = RawUtils.bayer01_2_rgb01(pred_bayer_01.cpu().numpy(), gamma=2.2, wb_gain=meta.wb_gain, CCM=meta.CCM)
                            labels_test = (labels_test*255.0).astype(np.uint8)
                            inputs_test = (inputs_test*255.0).astype(np.uint8)
                            outputs_test = (outputs_test*255.0).astype(np.uint8)
                            cv2.imwrite("gt_rgb_from_benchmark.bmp", labels_test)
                            cv2.imwrite("noisy_rgb_from_benchmark.bmp", inputs_test)
                            cv2.imwrite("denoised_rgb_from_benchmark.bmp", outputs_test)
                            pred_bayer_01
                            example_images.append({
                                'noisy_test': inputs_test,  # batch
                                'denoised_test': outputs_test,
                                'clean_test': labels_test
                            })
                            # break
                # # log metrics
                test_loss /= len(bm_loader)
                test_psnr /= len(bm_loader)
                print(f'Epoch: {epoch}, Test Loss: {test_loss:.6f}, Test PSNR: {test_psnr:.2f}')
                writer.add_scalar('test_loss', test_loss, step)
                writer.add_scalar('test_psnr', test_psnr, step)

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
                save_checkpoint(top_models, model, optimizer, epoch, step, test_psnr, args.model_dir)
                

def save_checkpoint(top_models, model, optimizer, epoch, step, psnr, model_dir):
    os.makedirs(model_dir, exist_ok=True)
                     
    if len(top_models) < 10 or psnr > min(top_models.keys()):
        # Remove the worst model if we already have 10
        if len(top_models) >= 10:
            worst_psnr = min(top_models.keys())
            os.remove(top_models[worst_psnr][0])
            del top_models[worst_psnr]        

        top_model_path = os.path.join(model_dir, 'top_models', f'top_model_psnr_{psnr:.2f}_step_{step}.pth')
        torch.save({
            'epoch':epoch,
            'step':step, 
            'state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'psnr':psnr}, 
            top_model_path)
        top_models[psnr] = (top_model_path, step)

        # # Save metadata about top models
        # with open(os.path.join(model_dir, 'top_models', 'top_models_info.txt'), 'w') as f:
        #     for psnr, (path, step) in sorted(top_models.items(), reverse=True):
        #         f.write(f"PSNR: {psnr:.2f}, Step: {step}, Path: {path}\n")   


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