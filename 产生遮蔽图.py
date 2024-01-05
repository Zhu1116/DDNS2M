import numpy as np
import matplotlib.pyplot as plt
import torch
import scipy.io

def random_mask(image, mask_ratio):
    # 随机生成 mask
    mask = np.random.rand(*image.shape) > mask_ratio
    binary_mask = mask.astype(int)
    # 将 mask 应用到图像上
    masked_image = np.where(mask, image, np.nan)
    return masked_image, binary_mask

# mat = scipy.io.loadmat('exp/datasets/Pavia.mat')
# img = mat['pavia']
img = np.load('exp/datasets/pavia.npy')

# 设置遮蔽比例（这里设置为 0.3 表示遮蔽 30% 的像素）
mask_ratio = 0.5

# 生成随机 mask
masked_image, mask = random_mask(img, mask_ratio)

# # 显示原始图像和带有随机 mask 的图像
# fig, axs = plt.subplots(1, 2, figsize=(12, 6))
#
# axs[0].imshow(img[:, :, 0], cmap='viridis')
# axs[0].set_title('Original Image')
#
# axs[1].imshow(mask[:, :, 0], cmap='viridis')
# axs[1].set_title('Image with Random Mask')
#
# plt.show()

np.save("exp/datasets/mask.npy", mask)