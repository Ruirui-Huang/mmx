import cv2
import numpy as np
import math
import random
import os.path as osp
import onnxruntime as ort
from tqdm import tqdm
from argparse import ArgumentParser
from .preprocess import Preprocess
from .decoder import Decoder
from .cv2_nms import non_max_suppression
from .utils import path_to_list
from .show_result import imshow_det


def onnx_od(path_imgs, args, out_dir=None):
    """ONNX前向，用于目标检测推理
    Args:
    Returns: 推理结果，及渲染之后的图片
    """
    assert args["Class_show"], print("class_show为空！")
    classes = args["Class_show"]["classes"]
    is_show = args["Class_show"]["is_show"]
    num_classes = args["Num_classes"]
    assert num_classes == len(classes) and  num_classes == len(is_show), print("类别长度不一致！")
    colors=[[random.randint(0, 255) for _ in range(3)] for _ in range(num_classes)]
    preprocessor = Preprocess()
    # 加载引擎
    sess = ort.InferenceSession(args["Path_onnx"], providers=['CUDAExecutionProvider'])
    decoder = Decoder(model_type=args["Model_type"], model_only=True)
    input_name = sess.get_inputs()[0].name
    input_size = sess.get_inputs()[0].shape[-2:]
    files = path_to_list(path_imgs)
    result = dict()
    for file in tqdm(files): 
        image = cv2.imread(file)
        H, W = image.shape[:2]
        # 数据预处理
        img, (ratio_w, ratio_h), padding_list = preprocessor(image, input_size)
        # 推理
        features = sess.run(None, {input_name: img})
        # 后处理
        decoder_outputs = decoder(
            features,
            args["Score_thr"],
            num_labels=num_classes,
            anchors=args["Anchors"])

        if len(decoder_outputs[0]) == 0: continue
        
        bboxes, _, labels = non_max_suppression(
            *decoder_outputs, 
            args["Score_thr"],
            args["Box_thr"],)

        pre_label = imshow_det(
            file, bboxes, labels,             
            save_path=out_dir, 
            is_show=is_show, 
            classes=classes, 
            scale_factor=(ratio_w, ratio_h),
            padding_list=padding_list)

        result[file] = pre_label

    return result

if __name__ == '__main__':
    args = {
        "Path_onnx": "./model.onnx",
        "Model_type": "yolov5",
        "Num_classes": 3,      
        "Score_thr": 0.3,
        "Box_thr": 0.65,     
        "Anchors": [[(9, 3), (6, 16), (26, 8)], [(15, 40), (32, 73), (63, 130)], [(91, 99), (190, 182),(339, 276)]],      
        "Class_show": {
                "classes": ['Person', 'Plate', 'Car'], 
                "is_show": [1, 1, 1]
        },
        "Parent": ["Car", None, None]         
    }
    onnx_od("./data", args, out_dir='./show')