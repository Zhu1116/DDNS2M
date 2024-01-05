import imgvision as iv
import torch
import numpy as np

def data_transform(X):
    return 2 * X - 1.0

def inverse_data_transform(X):
    return torch.clamp((X + 1.0) / 2.0, 0.0, 1.0)

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
