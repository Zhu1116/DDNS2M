from osgeo import gdal
import numpy as np


def crop_image(input_path, output_shape, x_start, y_start):
    # 打开 TIF 文件
    input_ds = gdal.Open(input_path, gdal.GA_ReadOnly)

    if input_ds is None:
        print("无法打开输入文件")
        return

    # 读取图像数据
    input_array = input_ds.ReadAsArray()

    # 裁剪图像
    cropped_array = input_array[:,x_start:x_start + output_shape[0] , y_start:y_start + output_shape[1]]

    return cropped_array


def get_crop_image():
    # 输入文件路径
    input_file_path = "exp/datasets/dc.tif"

    # 裁剪参数
    x_start = 570
    y_start = 20
    output_shape = (256, 256, 191)

    # 执行裁剪
    cropped_data = crop_image(input_file_path, output_shape, x_start, y_start)

    cropped_data = np.transpose(cropped_data, (1, 2, 0))
    max_val = np.max(cropped_data)
    min_val = np.min(cropped_data)
    cropped_data = (cropped_data - min_val) * 1.0 / max_val
    cropped_data = np.clip(cropped_data, 0., 1.)

    # 打印裁剪后的数组形状
    print("裁剪后的数组形状:", cropped_data.shape)

    # 保存为 NumPy 数组
    np.save("exp/datasets/dc.npy", cropped_data)

get_crop_image()


import imgvision as iv
# import matplotlib.pyplot as plt
# file_path = 'exp/datasets/dc.npy'
# img = np.load(file_path)
# print(img.shape)
#
# # convertor = iv.spectra(band = np.arange(400,2310 ,10))
# # rgb = convertor.space(img)
# # print(rgb.shape)
#
# img2 = np.concatenate((img[:,:,60:61], img[:,:,27:28], img[:,:,17:18]), axis=2)
# # max_val = np.max(img2)
# # min_val = np.min(img2)
# # img2 = (img2 - min_val) * 1.0 / max_val
# # print(img2.shape)
#
# plt.imshow(img2)
# plt.show()