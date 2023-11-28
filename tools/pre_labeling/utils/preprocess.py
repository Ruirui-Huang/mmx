# -*- coding: utf-8 -*-
import cv2
import numpy as np

class Preprocess:
    """默认数据预处理方式为强行缩放到指定大小
    fixed_scale (int): 值为0，图像直接缩放
                       值为1，图像保持原图宽高比缩放，居中对齐
                       值为2，图像保持原图宽高比缩放，底部对齐，左右居中
                       值为3，图像保持原图宽高比缩放，顶部对齐，左对齐
    color_space (str): "rgb"，"bgr"
    """
    def __init__(self, fixed_scale=0, color_space="rgb"):
        self.mean = np.array([0, 0, 0], dtype=np.float32).reshape((3, 1, 1))
        self.std = np.array([255, 255, 255], dtype=np.float32).reshape((3, 1, 1))
        self.fixed_scale = fixed_scale
        self.is_rgb = True if color_space == "rgb" else False

    def __call__(self, image, new_size):
        """数据预处理
        Args: 
            image (ndarray): 图片数据
            new_size (list): onnx输入大小 H, W
        """
        assert 0 <= self.fixed_scale <=3, print(f"缩放方式{self.fixed_scale}不支持！")
        height, width = image.shape[:2]
        height_new, width_new = new_size
        padding_list = [0]*4
        scale_factor = [new_size[0] / height, new_size[1] / width]
        if self.fixed_scale == 0:
            image = cv2.resize(image, (width_new, height_new))
        else:
            ratio = min(scale_factor)
            no_pad_shape = (int(round(height*ratio)),int(round(width*ratio)))
            padding_h, padding_w = [height_new - no_pad_shape[0], width_new - no_pad_shape[1]]
            if new_size != no_pad_shape:
                image = cv2.resize(image, (no_pad_shape[1], no_pad_shape[0]))
            
            scale_factor = (no_pad_shape[1]/width, no_pad_shape[0]/height)

            if self.fixed_scale == 1:
                top_padding, left_padding = int(round(padding_h // 2 - 0.1)), int(round(padding_w // 2 - 0.1))
                bottom_padding = padding_h - top_padding
                right_padding = padding_w - left_padding
            elif self.fixed_scale == 2:
                bottom_padding = 0
                left_padding = int(round(padding_w // 2 - 0.1))
                top_padding = padding_h - bottom_padding
                right_padding = padding_w - left_padding
            elif self.fixed_scale == 3:
                top_padding, left_padding = 0, 0
                bottom_padding = padding_h - top_padding
                right_padding = padding_w - left_padding

            padding_list = [
                top_padding, bottom_padding, left_padding, right_padding
            ]
            if top_padding != 0 or bottom_padding !=0 or left_padding !=0 or right_padding !=0:
                image = cv2.copyMakeBorder(
                    image,
                    top_padding,
                    bottom_padding,
                    left_padding,
                    right_padding,
                    cv2.BORDER_CONSTANT,
                    value=(114, 114, 114)
                )
        image = np.ascontiguousarray(image.transpose(2, 0, 1))
        image = image.astype(np.float32)
        image -= self.mean
        image /= self.std
        return image[np.newaxis], scale_factor, padding_list
