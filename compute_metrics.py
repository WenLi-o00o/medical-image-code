# _*_ coding:utf-8 _*_

import numpy as np
from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics import mean_squared_error as mse
from skimage.measure import compare_psnr as psnr
from skimage.measure import compare_ssim as ssim

true = np.load("/mnt/md0/wen/tiff_png_deform_data/data_for_test/p51/MR_npy.npy")
predict = np.load("/home/s195204/image-segmentation-keras-master/results/results/UNet-only-CT/results/result_51.npy")

total_mae = 0
total_mse = 0
total_psnr = 0
total_ssim = 0
for i in range(len(true)):
    mae_result0 = mae(true[i,:,:], predict[i,:,:])
    mse_result0 = mse(true[i,:,:], predict[i,:,:])
    psnr_result0 = psnr(true[i,:,:], predict[i,:,:])
    ssim_result0 = ssim(true[i,:,:], predict[i,:,:])
    total_mae += mae_result0
    total_mse += mse_result0
    total_psnr += psnr_result0
    total_ssim += ssim_result0

mae_result = total_mae / len(true)
mse_result = total_mse / len(true)
ssim_result = total_ssim / len(true)
psnr_result = total_psnr / len(true)
print ('MAE:', mae_result)
print('MSE:', mse_result)
print('PSNR:', psnr_result)
print('SSIM:', ssim_result)