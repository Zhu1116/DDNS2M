# import matplotlib.pyplot as plt
# import numpy as np
#
# # 设置长方形的高和宽
# height = 20  # 长方形的高度
# width = 1    # 长方形的宽度
#
# # 创建一个黑色的长方形
# rectangle = plt.Rectangle((0, 0), width, height, color='black')
#
# # 创建一个图形并添加长方形
# fig, ax = plt.subplots()
# ax.add_patch(rectangle)
#
# # 生成随机的白色遮挡
# num_bands = 10  # 白色遮挡的数量
# band_height = 0.5  # 遮挡的高度
#
# for _ in range(num_bands):
#     band_position = np.random.uniform(0, height - band_height)  # 遮挡的位置
#
#     # 创建白色遮挡
#     band = plt.Rectangle((0, band_position), width, band_height, color='white')
#     ax.add_patch(band)
#
# # 设置坐标轴
# ax.set_xlim(0, width)
# ax.set_ylim(0, height)
#
# # 隐藏坐标轴
# ax.axis('off')
#
# # 显示图形
# plt.show()


import cv2
import numpy as np

# 设置长方形的高和宽
height = 100  # 长方形的高度
width = 10    # 长方形的宽度

# 创建一个黑色的图像
image = np.zeros((height, width, 3), dtype=np.uint8)

# 生成随机的白色遮挡
num_bands = 50  # 白色遮挡的数量
band_height = 1  # 遮挡的高度，设为1确保覆盖整个宽度

for _ in range(num_bands):
    band_position = np.random.randint(0, height - band_height)  # 遮挡的位置

    # 添加白色遮挡
    image[band_position:band_position+band_height, :, :] = [255, 255, 255]

cv2.imwrite('spectral.png', image)
# 显示图像
cv2.imshow('Result', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
