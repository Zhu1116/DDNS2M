import scipy.io as scio
import os
import numpy as np

img_clean = np.load('exp/datasets/pavia.npy')
mask = np.load('exp/datasets/mask.npy')


scio.savemat(
    'exp/datasets/pavia_inpainting.mat',
    {'img_clean':img_clean, 'mask':mask}
)