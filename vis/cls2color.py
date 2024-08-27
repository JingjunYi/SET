import os
import numpy as np
from PIL import Image
from tqdm import tqdm

# Cityscapes颜色映射字典
cityscapes_color_mapping = {
    0: (128, 64, 128),   # road
    1: (244, 35, 232),   # sidewalk
    2: (70, 70, 70),     # building
    3: (102, 102, 156),  # wall
    4: (190, 153, 153),  # fence
    5: (153, 153, 153),  # pole
    6: (250, 170, 30),   # traffic light
    7: (220, 220, 0),    # traffic sign
    8: (107, 142, 35),   # vegetation
    9: (152, 251, 152),  # terrain
    10: (70, 130, 180),  # sky
    11: (220, 20, 60),   # person
    12: (255, 0, 0),     # rider
    13: (0, 0, 142),     # car
    14: (0, 0, 70),      # truck
    15: (0, 60, 100),    # bus
    16: (0, 80, 100),    # train
    17: (0, 0, 230),     # motorcycle
    18: (119, 11, 32),   # bicycle
    19: (0, 0, 0)        # void
}

# 定义函数对单张mask进行上色
def colorize_mask(mask_path, save_path):
    # 加载mask图像
    mask = np.array(Image.open(mask_path))

    # 生成着色后的mask图像
    height, width = mask.shape
    colored_mask = np.zeros((height, width, 3), dtype=np.uint8)
    for i in range(height):
        for j in range(width):
            category = mask[i, j]
            colored_mask[i, j] = cityscapes_color_mapping.get(category, (0, 0, 0))  # 使用字典进行颜色映射

    # 保存着色后的mask图像
    Image.fromarray(colored_mask).save(save_path)

# 定义函数批量处理文件夹中的mask图像
def batch_colorize_masks(input_folder, output_folder):
    # 确保输出文件夹存在
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 遍历输入文件夹中的所有mask图像文件
    for filename in tqdm(os.listdir(input_folder)):
        if filename.endswith('.png') or filename.endswith('.jpg'):  # 假设所有的mask图像文件都是PNG格式或JPEG格式
            mask_path = os.path.join(input_folder, filename)
            save_path = os.path.join(output_folder, filename)
            colorize_mask(mask_path, save_path)

# 执行批量处理
input_folder = 'G:/MM2024/SET-main/vis/nips-rebuttal/gt'  # 输入mask图像所在的文件夹路径
output_folder = 'G:/MM2024/SET-main/vis/nips-rebuttal/gt_c'  # 输出着色后mask图像的文件夹
# os.mkdir(output_folder)
batch_colorize_masks(input_folder, output_folder)
