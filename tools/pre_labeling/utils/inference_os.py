# -*- coding: utf-8 -*-
import cv2
import numpy as np
import os.path as osp
import onnxruntime as ort
from .preprocess import Preprocess

def inference_os(img, path_model):
    """ONNX前向
    Args:
        img (str | ndarray): 图片路径 or 图片数据
        path_model (str): onnx路径
    """
    if isinstance(img, str): 
        if osp.splitext(img)[-1] != '.jpg': return
        image = cv2.imread(img)
    else:
        image = img
    
    preprocessor = Preprocess()
    assert osp.exists(path_model), print(f"{path_model}不存在！")
    # 加载引擎
    sess = ort.InferenceSession(path_model, providers=['CUDAExecutionProvider'])
    # 数据预处理
    input_size = sess.get_inputs()[0].shape[-2:]
    H, W, _ = image.shape
    input_data, (_, _), _ = preprocessor(image, input_size)
    input_name = sess.get_inputs()[0].name
    # 推理
    outputs = sess.run(output_names=None, input_feed={input_name: input_data})[0][0] # C, W, H
    # 后处理
    outputs = outputs.transpose(1, 2, 0) # W, H, C
    outputs = cv2.resize(outputs, [W, H]) # H, W, C
    return np.argmax(outputs, 2) # H, W
