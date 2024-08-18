# -*- coding: utf-8 -*-
# @Date    : 2024-08-18 20:55:17
# @Author  : huang_rui

TESTDATA = ["PASCAL_VOC_2012"]
TRAINDATA = ["ADE20K", "BDD", "Cityscapes", "COCOPanoptic", "IDD", "MapillaryVistasPublic", "SUNRGBD", "WildDash", "PASCAL_Context","KITTI","Camvid", "ScanNet"]

root_path_coco = "D:/Data_pub/00_coco2017/labels/val2017"

import os
import glob
from PIL import Image
import numpy as np
from multiprocessing import Pool
import matplotlib.pyplot as plt
from tqdm import tqdm

def process_image(filepath):
    try:
        # 使用PIL读取图片
        img = Image.open(filepath)
        # 将图片转换为numpy数组
        pixels = np.array(img)
        # 将像素值展平
        return pixels.flatten()
    except Exception as e:
        print(f"Error processing {filepath}: {e}")
        return []

def analyze_pixels(pixel_lists):
    # 将所有像素值合并成一个大列表
    all_pixels = np.concatenate(pixel_lists)
    # 过滤像素值为0的情况
    filtered_pixels = all_pixels[all_pixels != 0]
    # 使用numpy进行统计分析
    unique, counts = np.unique(filtered_pixels, return_counts=True)
    pixel_distribution = dict(zip(unique, counts))
    print(pixel_distribution)
    return pixel_distribution

def plot_pixel_distribution(pixel_distribution):
    # 绘制直方图
    plt.figure(figsize=(10, 6))
    plt.bar(pixel_distribution.keys(), pixel_distribution.values(), color='blue', alpha=0.7)
    plt.title('Pixel Value Distribution')
    plt.xlabel('Pixel Value')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.savefig("pixel_distribution.png")
    plt.show()

def main(directory):
    # 获取所有PNG文件的路径
    png_files = glob.glob(os.path.join(directory, "*.png"))
    
    # 创建进程池
    with Pool(processes=os.cpu_count()) as pool:
        # 并行处理所有图片，并添加进度条
        results = list(tqdm(pool.imap(process_image, png_files), total=len(png_files)))
    
    # 分析像素分布
    pixel_distribution = analyze_pixels(results)
    
    # 绘制分布结果图
    plot_pixel_distribution(pixel_distribution)

if __name__ == "__main__":
    main(root_path_coco)