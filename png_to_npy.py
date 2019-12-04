#_*_ coding:utf-8 _*_

import scipy.misc
#import cv2
import numpy as np
import os
#from skimage.color import rgb2gray

path1 = "/mnt/md0/wen/tiff_png_deform_data/data_for_test/p51/MR-deformed/"
path2 = "/mnt/md0/wen/tiff_png_deform_data/data_for_test/p51/"

data = os.listdir(path1)
#print(data)
data.sort()
#print(data)
for i in range(len(data)):
    data[i] = data[i].split('.')
    data[i][0] = int(data[i][0])
data.sort()
for i in range(len(data)):
    data[i][0] = str(data[i][0])
    data[i] = data[i][0] + '.' + data[i][1]

print(data)

empty_npy = np.ndarray((len(data),256,256),dtype='uint8')
 
for i in range(len(data)):
    img = scipy.misc.imread(os.path.join(path1,data[i]))
    if len(img.shape)>2:
        #a = img[:,:,:-1]
        #img = rgb2gray(img)
        img = img[:,:,0]
    img = np.array(img,dtype='uint8')
    empty_npy[i] = img

np.save(os.path.join(path2,'DMR_npy.npy'),empty_npy)
print('Finished!')