import torch
from tqdm import tqdm
import torchvision.utils as tvu
import torchvision
import os
import numpy as np
from skimage.metrics import structural_similarity as compare_ssim
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
import wandb
import matplotlib.pyplot as plt
import imgvision as iv

class_num = 951


def compute_alpha(beta, t):
    beta = torch.cat([torch.zeros(1).to(beta.device), beta], dim=0)
    a = (1 - beta).cumprod(dim=0).index_select(0, t + 1).view(-1, 1, 1, 1)
    return a

def inverse_data_transform(x):
    x = (x + 1.0) / 2.0
    return torch.clamp(x, 0.0, 1.0)

def ddnm_diffusion(x, model, b, eta, A_funcs, y, cls_fn=None, classes=None, config=None, logger=None, img_clean=None):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    if 1 > 0:

        # setup iteration variables
        skip = config.diffusion.num_diffusion_timesteps//config.time_travel.T_sampling
        n = x.size(0)
        x0_preds = []
        xs = [x]

        # generate time schedule
        times = get_schedule_jump(config.time_travel.T_sampling, 
                               config.time_travel.travel_length, 
                               config.time_travel.travel_repeat,
                              )
        time_pairs = list(zip(times[:-1], times[1:]))

        # 记录最好结果
        x0_psnr_best = 0
        xt_psnr_best = 0
        x0_best = None
        
        # reverse diffusion sampling
        for i, j in tqdm(time_pairs):
            i, j = i*skip, j*skip
            if j<0: j=-1 

            if j < i: # normal sampling
                with torch.no_grad():
                    t = (torch.ones(n) * i).to(x.device)
                    next_t = (torch.ones(n) * j).to(x.device)
                    at = compute_alpha(b, t.long())
                    at_next = compute_alpha(b, next_t.long())
                    xt = xs[-1].to(device)
                    avg = np.array(0)

                x0_t, step, best_, psnr_best_ \
                    = model.optimize(xt.squeeze().permute(1, 2, 0).detach().cpu().numpy(),
                                     img_clean.squeeze().permute(1, 2, 0).detach().cpu().numpy(),
                                     at, None, 0, logger, avg, True)
                x0_t = torch.from_numpy(x0_t).permute(2, 0, 1).unsqueeze(0).to(device)
                x0_t = x0_t * 2 - 1.0

                with torch.no_grad():
                    et = (xt - at.sqrt() * x0_t) / (1 - at).sqrt()


                    x0_t_hat = x0_t - A_funcs.A_pinv(
                        A_funcs.A(x0_t.reshape(x0_t.size(0), -1)) - y.reshape(y.size(0), -1)
                    ).reshape(*x0_t.size())

                    # x0_t_hat = A_funcs.A_pinv(y.reshape(y.size(0), -1)).reshape(*x0_t.size())

                    c1 = (1 - at_next).sqrt() * eta
                    c2 = (1 - at_next).sqrt() * ((1 - eta ** 2) ** 0.5)
                    xt_next = at_next.sqrt() * x0_t_hat + c1 * torch.randn_like(x0_t) + c2 * et

                    x0_preds.append(x0_t.to('cpu'))
                    xs.append(xt_next.to('cpu'))

                    # 后加的
                    x0_t = torch.clamp((x0_t + 1.0) / 2.0, 0.0, 1.0)
                    avg = x0_t[0, :, :, :].permute(1, 2, 0).cpu().numpy()
                    from runners.com_psnr import quality
                    x0_psnr = quality(x0_t.squeeze().cpu().permute(1, 2, 0).numpy(),
                                   img_clean.squeeze().permute(1, 2, 0).numpy())

                    xt_psnr = quality(torch.clamp((xt_next[0]+1.0)/2.0,0.0,1.0).cpu().permute(1, 2, 0).numpy(),
                                   img_clean.squeeze().permute(1, 2, 0).numpy())

                    if x0_psnr_best < x0_psnr:
                        x0_psnr_best = x0_psnr
                        x0_best = x0_t.squeeze().cpu().permute(1, 2, 0).numpy()
                    if xt_psnr_best < xt_psnr:
                        xt_psnr_best = xt_psnr
                    print('iteration: %05d, x0_psnr: %.4f, x0_psnr_best: %.4f, xt_psnr: %.4f, xt_psnr_best: %.4f'
                                % (i, x0_psnr, x0_psnr_best, xt_psnr, xt_psnr_best))
            else: # time-travel back
                next_t = (torch.ones(n) * j).to(x.device)
                at_next = compute_alpha(b, next_t.long())
                x0_t = x0_preds[-1].to('cuda')
                
                xt_next = at_next.sqrt() * x0_t + torch.randn_like(x0_t) * (1 - at_next).sqrt()

                xs.append(xt_next.to('cpu'))

    return xs[-1], [x0_preds[-1]]

def ddnm_plus_diffusion(x, model, b, eta, A_funcs, y, sigma_y, cls_fn=None, classes=None, config=None, logger=None, img_clean=None, save_dir=None):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    clean_img = img_clean.squeeze().permute(1, 2, 0).numpy()

    # setup iteration variables
    skip = config.diffusion.num_diffusion_timesteps//config.time_travel.T_sampling
    n = x.size(0)
    x0_preds = []
    xs = [x]

    # generate time schedule
    times = get_schedule_jump(config.time_travel.T_sampling,
                           config.time_travel.travel_length,
                           config.time_travel.travel_repeat,
                          )
    time_pairs = list(zip(times[:-1], times[1:]))

    # 记录最好结果
    psnr_best = 0
    xt_psnr_best = 0
    ssim_best = 0
    xt_ssim_best = 0
    sam_best = 1000
    xt_sam_best = 1000
    loss = 0
    iii = 0

    # reverse diffusion sampling
    for i, j in tqdm(time_pairs):
        iii += 1
        i, j = i*skip, j*skip
        if j<0: j=-1

        if j < i: # normal sampling
            with torch.no_grad():
                t = (torch.ones(n) * i).to(x.device)
                next_t = (torch.ones(n) * j).to(x.device)
                at = compute_alpha(b, t.long())
                at_next = compute_alpha(b, next_t.long())
                if iii == 1:
                    xt = x.to(device)
                    avg = np.array(0)
                else:
                    xt = xt_next.to(x.device)
                    avg = x0_t[0, :, :, :].permute(1, 2, 0).cpu().numpy()

            update = False if iii < (len(time_pairs) / 2) else True
            if update:
                x0_t, loss, best_, psnr_best_ \
                    = model.optimize(xt.squeeze().permute(1, 2, 0).detach().cpu().numpy(),
                                     img_clean.squeeze().permute(1, 2, 0).detach().cpu().numpy(),
                                     at, None, 0, logger, avg, True)
                x0_t = torch.from_numpy(x0_t).permute(2, 0, 1).unsqueeze(0).to(device)
                x0_t = x0_t * 2 - 1.0
            else:
                x0_t = xt

            et = (xt - at.sqrt() * x0_t) / (1 - at).sqrt()

            with torch.no_grad():
                sigma_t = (1 - at_next).sqrt()[0, 0, 0, 0]

                # Eq. 17
                x0_t_hat = x0_t - A_funcs.Lambda(A_funcs.A_pinv(
                    A_funcs.A(x0_t.reshape(x0_t.size(0), -1)) - y.reshape(y.size(0), -1)
                ).reshape(x0_t.size(0), -1), at_next.sqrt()[0, 0, 0, 0], sigma_y, sigma_t, eta).reshape(*x0_t.size())

                # Eq. 51
                xt_next = at_next.sqrt() * x0_t_hat + A_funcs.Lambda_noise(
                    torch.randn_like(x0_t).reshape(x0_t.size(0), -1),
                    at_next.sqrt()[0, 0, 0, 0], sigma_y, sigma_t, eta, et.reshape(et.size(0), -1)).reshape(*x0_t.size())

                # x0_preds.append(x0_t.to('cpu')) # 超内存了
                # xs.append(xt_next.to('cpu'))

                x0_t = torch.clamp((x0_t + 1.0) / 2.0, 0.0, 1.0)
                # 评估
                if update:
                    x_0 = x0_t.squeeze().cpu().permute(1, 2, 0).numpy()
                    x_t = torch.clamp((xt_next + 1.0) / 2.0, 0.0, 1.0)
                    x_t = x_t.squeeze().cpu().permute(1, 2, 0).numpy()

                    # psnr
                    psnr = compare_psnr(clean_img, x_0)
                    xt_psnr = compare_psnr(clean_img, x_t)
                    if psnr_best < psnr:
                        psnr_best = psnr
                        # plt.imsave('experiment/' + save_dir + '/best/x0_i_%03d_%.3f.png' % (i, psnr), to_rgb(x_0))
                        plt.imsave('experiment/' + save_dir + '/best/x0_i_%03d_%.3f.png' % (i, psnr),
                                   np.concatenate((x_0[:, :, 60:61], x_0[:, :, 27:28], x_0[:, :, 17:18]), axis=2))
                    if xt_psnr_best < xt_psnr:
                        xt_psnr_best = xt_psnr

                    # ssim
                    ssim = compare_ssim(clean_img, x_0, multichannel=True, data_range=1)
                    xt_ssim = compare_ssim(clean_img, x_t, multichannel=True, data_range=1)
                    if ssim_best < ssim:
                        ssim_best = ssim
                    if xt_ssim_best < xt_ssim:
                        xt_ssim_best = xt_ssim

                    # sam
                    sam = SAM(clean_img, x_0)
                    xt_sam = SAM(clean_img, x_t)
                    if sam_best > sam:
                        sam_best = sam
                    if xt_sam_best > xt_sam:
                        xt_sam_best = xt_sam

                    print('iteration: %03d, loss: %.4f, x0_psnr: %.3f, x0_psnr_best: %.3f, xt_psnr: %.3f, xt_psnr_best: %.3f'
                        % (i, loss, psnr, psnr_best, xt_psnr, xt_psnr_best))

                    wandb.log({
                        "loss": loss,
                        "psnr": psnr,
                        "psnr best": psnr_best,
                        "xt psnr": xt_psnr,
                        "xt psnr best": xt_psnr_best,
                        "ssim": ssim,
                        "ssim best": ssim_best,
                        "xt ssim": xt_ssim,
                        "xt ssim best": xt_ssim_best,
                        "sam": sam,
                        "sam best": sam_best,
                        "xt sam": xt_sam,
                        "xt sam best": xt_sam_best,
                    })
                    if i % 5 == 0:
                        # plt.imsave('experiment/'+save_dir+'/result/x0_i_%03d_%.3f.png' % (i, psnr), to_rgb(x_0))
                        plt.imsave('experiment/' + save_dir + '/result/x0_i_%03d_%.3f.png' % (i, psnr),
                                   np.concatenate((x_0[:, :, 60:61], x_0[:, :, 27:28], x_0[:, :, 17:18]), axis=2))

        else: # time-travel back
            next_t = (torch.ones(n) * j).to(x.device)
            at_next = compute_alpha(b, next_t.long())
            x0_t = x0_preds[-1].to('cuda')

            xt_next = at_next.sqrt() * x0_t + torch.randn_like(x0_t) * (1 - at_next).sqrt()

            xs.append(xt_next.to('cpu'))

#             #ablation
#             if i%50==0:
#                 os.makedirs('/userhome/wyh/ddnm/debug/x0t', exist_ok=True)
#                 tvu.save_image(
#                     inverse_data_transform(x0_t[0]),
#                     os.path.join('/userhome/wyh/ddnm/debug/x0t', f"x0_t_{i}.png")
#                 )

#                 os.makedirs('/userhome/wyh/ddnm/debug/x0_t_hat', exist_ok=True)
#                 tvu.save_image(
#                     inverse_data_transform(x0_t_hat[0]),
#                     os.path.join('/userhome/wyh/ddnm/debug/x0_t_hat', f"x0_t_hat_{i}.png")
#                 )

#                 os.makedirs('/userhome/wyh/ddnm/debug/xt_next', exist_ok=True)
#                 tvu.save_image(
#                     inverse_data_transform(xt_next[0]),
#                     os.path.join('/userhome/wyh/ddnm/debug/xt_next', f"xt_next_{i}.png")
#                 )


    return x0_t.to('cpu'), xt_next.to('cpu')

# form RePaint
def get_schedule_jump(T_sampling, travel_length, travel_repeat):

    jumps = {}
    for j in range(0, T_sampling - travel_length, travel_length):
        jumps[j] = travel_repeat - 1

    t = T_sampling
    ts = []

    while t >= 1:
        t = t-1
        ts.append(t)

        if jumps.get(t, 0) > 0:
            jumps[t] = jumps[t] - 1
            for _ in range(travel_length):
                t = t + 1
                ts.append(t)

    ts.append(-1)

    _check_times(ts, -1, T_sampling)

    return ts

def _check_times(times, t_0, T_sampling):
    # Check end
    assert times[0] > times[1], (times[0], times[1])

    # Check beginning
    assert times[-1] == -1, times[-1]

    # Steplength = 1
    for t_last, t_cur in zip(times[:-1], times[1:]):
        assert abs(t_last - t_cur) == 1, (t_last, t_cur)

    # Value range
    for t in times:
        assert t >= t_0, (t, t_0)
        assert t <= T_sampling, (t, T_sampling)

def to_rgb(X):
    convertor = iv.spectra()
    rgb = convertor.space(X)
    return rgb

def SAM(x_true, x_pred):
    assert x_true.ndim == 3 and x_true.shape == x_pred.shape
    dot_sum = np.sum(x_true * x_pred, axis=2)
    norm_true = np.linalg.norm(x_true, axis=2)
    norm_pred = np.linalg.norm(x_pred, axis=2)

    res = np.arccos(dot_sum / norm_pred / norm_true)
    is_nan = np.nonzero(np.isnan(res))
    for (x, y) in zip(is_nan[0], is_nan[1]):
        res[x, y] = 0

    sam = np.mean(res)
    return sam