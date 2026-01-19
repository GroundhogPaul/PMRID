''' study API: the torchvision.transforms.ColorJitter by compared with my own code'''
import utilBasicRun
import numpy as np
import torch
import cv2
import torchvision.transforms as tvtransforms

def SaveJpgFromTensorChHW(tensor, sSavePath):
    img = (tensor.permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)
    cv2.imwrite(sSavePath, img)

# ---------- test on rgb ---------- #
sImgIn = r"D:\users\xiaoyaopan\PxyAI\RawDnsTimBrooks\BabySmile.jpg"
imgInU8 = cv2.imread(sImgIn)
imgIn = imgInU8.astype(float)
imgIn_01 = imgIn / 255.0

tensorIn = torch.from_numpy(imgIn_01).float() # [H, W, C]
tensorIn = tensorIn.permute(2, 0, 1)  # to [C, H, W]

# ----- case 1: does 'brightness' clip? ----- #
# channel_means = tensorIn.mean(dim=(1, 2))
# print(f"before: mean of 3 channel: {channel_means}, mean of total: {torch.mean(tensorIn):0.4f}")
# print(f"before: max: {torch.max(tensorIn):0.4f}, min: {torch.min(tensorIn):0.4f}")
# tensorJitter = tvtransforms.ColorJitter(brightness=(10.5, 10.5))(tensorIn)
# tensorJitter = tvtransforms.ColorJitter(brightness=(0.5, 0.5))(tensorIn)
# channel_means = tensorJitter.mean(dim=(1, 2))
# print(f"after: mean of 3 channel: {channel_means}, mean of tota: {torch.mean(tensorJitter):0.4f}")
# print(f"after: max: {torch.max(tensorJitter):0.4f}, min: {torch.min(tensorJitter):0.4f}")
# print(f"Jitter 中的brightness自带clip效果")

# ----- case 2: does 'contrast' clip? ----- #
# print(f"before: max: {torch.max(tensorIn):0.4f}, min: {torch.min(tensorIn):0.4f}")
# tensorJitter = tvtransforms.ColorJitter(contrast=(1.5, 1.5))(tensorIn)
# print(f"after: max: {torch.max(tensorJitter):0.4f}, min: {torch.min(tensorJitter):0.4f}")
# print(f"Jitter 中的contrast自带clip效果")

# ----- case 3: self method : normal mode ----- #
meanFactor = 0.2
contrastFactor = 15
tensorJitter = tvtransforms.ColorJitter(brightness=(meanFactor, meanFactor), contrast=(contrastFactor, contrastFactor))(tensorIn)
SaveJpgFromTensorChHW(tensorJitter, "ColorJitter.png")

tensorJitterMy = tensorIn * meanFactor
mean = torch.mean(tensorJitterMy)
tesnorJitterMy = tensorJitterMy + (tensorJitterMy - mean) * contrastFactor 
SaveJpgFromTensorChHW(tensorJitter, "ColorJitterMy.png")


# print(f"before: max: {torch.max(tensorIn):0.4f}, min: {torch.min(tensorIn):0.4f}")
# tensorJitter = tvtransforms.ColorJitter(brightness=(1.5, 1.5), contrast = (1.5, 1.5))(tensorIn)
# print(f"after: max: {torch.max(tensorJitter):0.4f}, min: {torch.min(tensorJitter):0.4f}")

# ----- save img ----- #
cv2.imwrite("JitterBefore.jpg", imgInU8)


# ---------- test on raw ---------- #
# sRawIn = "D:/image_database/SID/SID/Sony/long/00002_00_10s.ARW"

# import rawpy
# input_raw = rawpy.imread(sRawIn)

# white_level = input_raw.white_level
# black_level = input_raw.black_level_per_channel[0]
# bayer_pattern = input_raw.raw_pattern
# input_bayer = input_raw.raw_image.astype(np.float32)
# H, W = input_bayer.shape

# wb_gain = np.array([wb / 1024.0 for wb in input_raw.camera_whitebalance]),
# ccm = input_raw.rgb_xyz_matrix[:3, :]

# input_bayer_01 = input_bayer/ white_level
# input_bayer_01 = torch.from_numpy(input_bayer_01).unsqueeze(0)  # to [1, H, W] Tensor
# input_bayer_01 = tvtransforms.ColorJitter(brightness=(0.2, 1.2), contrast=(0.5, 1.5))(input_bayer_01)

# RawUtils.


