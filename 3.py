"""
    WDC数据，去噪
"""
import logging

import torch
import numpy as np
from runners.VS2M import VS2M
from functions.svd_operators import SRConv, Denoising
import scipy
import os
from tools import *
from tqdm import tqdm
import matplotlib.pyplot as plt
import wandb
from skimage.metrics import structural_similarity as compare_ssim
from skimage.metrics import peak_signal_noise_ratio as compare_psnr

# wandb.login(key="4067c8333229527e30a319c4395f4159d2bac099")
# run = wandb.init(project="DIP-project", group="experiment1", name="denoising0.1")

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# 预处理参数
# deg = 'sr_bicubic'
deg = 'denoising'
deg_scale = 4
sigma_0 = 0.2
# sigma_0 = 0

# 模型参数
rank = 10  # R的数量
beta = 0  # loss3的权重

# 数据参数
data_path = 'exp/datasets/dc.npy'
image_size = 256
channels = 191

# 训练参数
epoch_num = 800
lr = 0.0005

# 数据保存参数
save_dir = deg + str(sigma_0) + '_DS2DP'
save_interval = 5  # 每几轮保存一次图片


# 设置随机种子，确保结果一致
seed = 1234
torch.manual_seed(seed)
np.random.seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

torch.backends.cudnn.benchmark = True


# 初始化模型
model = VS2M(rank=rank,
             image_noisy=np.ones((image_size, image_size, channels)),
             image_clean=np.ones((image_size, image_size, channels)),
             beta=beta,
             num_iter=0,
             lr=lr)

# 初始化退化算法
A_funcs = None
if deg == 'sr_bicubic':
    factor = deg_scale
    def bicubic_kernel(x, a=-0.5):
        if abs(x) <= 1:
            return (a + 2) * abs(x) ** 3 - (a + 3) * abs(x) ** 2 + 1
        elif 1 < abs(x) and abs(x) < 2:
            return a * abs(x) ** 3 - 5 * a * abs(x) ** 2 + 8 * a * abs(x) - 4 * a
        else:
            return 0
    k = np.zeros((factor * 4))
    for i in range(factor * 4):
        x = (1 / factor) * (i - np.floor(factor * 4 / 2) + 0.5)
        k[i] = bicubic_kernel(x)
    k = k / np.sum(k)
    kernel = torch.from_numpy(k).float().to(device)
    A_funcs = SRConv(kernel / kernel.sum(), channels, image_size, device, stride=factor)
elif deg == 'denoising':
    A_funcs = Denoising(channels, image_size, device)


# 加载数据
# mat = scipy.io.loadmat(data_path)
# img_clean = torch.from_numpy(np.float32(mat['HSI'])).permute(2, 0, 1).unsqueeze(0)
img = np.load(data_path)
img_clean = torch.from_numpy(np.float32(img)).permute(2, 0, 1).unsqueeze(0)

x_orig = img_clean
x_orig = x_orig.to(device)
x_orig = data_transform(x_orig)

y = A_funcs.A(x_orig)
sigma_0 = 2 * sigma_0

b, hwc = y.size()
hw = hwc / channels
h = w = int(hw ** 0.5)
y = y.reshape((b, channels, h, w))
if sigma_0 > 0:
    y = y + torch.randn_like(y).to(device) * sigma_0
y = y.reshape((b, hwc))

Apy = A_funcs.A_pinv(y).view(y.shape[0], channels, image_size, image_size)

if not os.path.exists('experiment'):
    os.mkdir('experiment')
if not os.path.exists('experiment/'+save_dir):
    os.mkdir('experiment/'+save_dir)
result_save_dir = 'experiment/'+save_dir+'/result'
if not os.path.exists(result_save_dir):
    os.mkdir(result_save_dir)
best_save_dir = 'experiment/'+save_dir+'/best'
if not os.path.exists(best_save_dir):
    os.mkdir(best_save_dir)

# 保存为RGB图像
np_y = inverse_data_transform(Apy).squeeze().permute(1, 2, 0).detach().cpu().numpy()
rgb_y = np.concatenate((np_y[:,:,60:61], np_y[:,:,27:28], np_y[:,:,17:18]), axis=2)
np_x = inverse_data_transform(x_orig).squeeze().permute(1, 2, 0).detach().cpu().numpy()
rgb_x = np.concatenate((np_x[:,:,60:61], np_x[:,:,27:28], np_x[:,:,17:18]), axis=2)
plt.imsave('experiment/'+save_dir+'/Apy.png', rgb_y)
plt.imsave('experiment/'+save_dir+'/orig.png', rgb_x)

# 记录指标
psnr_best = 0.0
ssim_best = 0.0
sam_best = 1000.0

for epoch in tqdm(range(epoch_num)):
    img_pred, loss, _, _ = model.optimize(image_noisy=Apy.squeeze().permute(1, 2, 0).detach().cpu().numpy(),
                                       image_clean=img_clean.squeeze().permute(1, 2, 0).detach().cpu().numpy(),
                                       at=torch.tensor([[[[1.0]]]]),
                                       mask=None,
                                       iteration=0,
                                       logger=logging.getLogger(),
                                       avg=np.array(0),
                                       update=True)

    psnr = compare_psnr(img_clean.squeeze().permute(1, 2, 0).numpy(), img_pred)
    ssim = compare_ssim(img_clean.squeeze().permute(1, 2, 0).numpy(), img_pred, multichannel=True, data_range=1)
    sam = SAM(img_clean.squeeze().permute(1, 2, 0).numpy(), img_pred)
    if psnr > psnr_best:
        psnr_best = psnr
        best = np.concatenate((img_pred[:,:,60:61], img_pred[:,:,27:28], img_pred[:,:,17:18]), axis=2)
        plt.imsave(best_save_dir + '/epoch_%03d_%.3f.png' % (epoch, psnr_best), best)
    if ssim > ssim_best:
        ssim_best = ssim
    if sam < sam_best:
        sam_best = sam
    print('epoch: %03d, loss: %.4f, psnr: %.3f, psnr_best: %.3f, ssim: %.3f, ssim_best: %.3f, sam: %.3f, sam_best: %.3f'
          % (epoch, loss, psnr, psnr_best, ssim, ssim_best, sam, sam_best))

    # wandb.log({
    #     "loss": loss,
    #     "psnr": psnr,
    #     "psnr best": psnr_best,
    #     "ssim": ssim,
    #     "ssim best": ssim_best,
    #     "sam": sam,
    #     "sam best": sam_best
    # })

    if epoch % save_interval == 0:
        save = np.concatenate((img_pred[:,:,60:61], img_pred[:,:,27:28], img_pred[:,:,17:18]), axis=2)
        plt.imsave(result_save_dir+'/epoch_%03d_%.3f.png'%(epoch, psnr), save)

# run.finish()






# Apy_psnr = compare_psnr(img_clean.squeeze().permute(1, 2, 0).numpy(),
#                          Apy.squeeze().permute(1, 2, 0).detach().cpu().numpy())
# iv_psnr = get_psnr(img_clean.squeeze().permute(1, 2, 0).numpy(),
#                          Apy.squeeze().permute(1, 2, 0).detach().cpu().numpy())
# Apy_ssim = compare_ssim(img_clean.squeeze().permute(1, 2, 0).numpy(),
#                          Apy.squeeze().permute(1, 2, 0).detach().cpu().numpy(),
#                          multichannel=True, data_range=1)
# iv_ssim = get_ssim(img_clean.squeeze().permute(1, 2, 0).numpy(),
#                          Apy.squeeze().permute(1, 2, 0).detach().cpu().numpy())
# print('Apy_psnr: %.4f' % Apy_psnr)
# print('iv-psnr: %.4f' % iv_psnr)
# print('Apy_ssim: %.4f' % Apy_ssim)
# print('iv_ssim: %.4f' % iv_ssim)

# print('Apy_psnr: %.4f' % Apy_psnr)
# print('Apy_ssim: %.4f' % Apy_ssim)