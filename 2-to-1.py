# _*_ coding: utf-8 _*_

import numpy as np

imgs_test_CT = np.load("/mnt/md0/wen/tiff_png_deform_data/data_for_test/p51/CT_npy.npy")
imgs_test_DMR = np.load("/mnt/md0/wen/tiff_png_deform_data/data_for_test/p51/DMR_npy.npy")
imgs_test = np.ndarray((len(imgs_test_CT),256,256,2),dtype='uint8')
imgs_test[:,:,:,0] = imgs_test_CT
imgs_test[:,:,:,1] = imgs_test_DMR

np.save("/mnt/md0/wen/tiff_png_deform_data/data_for_test/p51/merged.npy",imgs_test)