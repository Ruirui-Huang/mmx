# -*- coding: utf-8 -*-
import cv2
import numpy as np
import math
import random
import glob
import os.path as osp
import caffe
import onnxruntime as ort
from tqdm import tqdm
from .preprocess import Preprocess
from .decoder import Decoder
from .nms import non_max_suppression
from .show_result import imshow_det


def inference(path_imgs, args, out_dir=None):
    """MODEL前向，用于目标检测推理
    Args:
        args (dict): model参数
        out_dir: 推理结果可视化存储路径
    """
    assert args["Class_show"], print("class_show为空！")
    classes = args["Class_show"]["classes"]
    is_show = args["Class_show"]["is_show"]
    exist_child = args["Class_show"]["exist_child"]
    num_classes = args["Num_classes"]
    assert num_classes == len(classes) and  \
        num_classes == len(is_show) and  \
            num_classes == len(exist_child), print("类别长度不一致！")
    if isinstance(path_imgs, str): path_imgs = glob.glob(osp.join(path_imgs, "*.jpg"))
    preprocessor = Preprocess(fixed_scale=1)
    decoder = Decoder(model_type=args["Model_type"], model_only=True)
    # 加载引擎
    if args["Model_type"] == "onnx":
        sess = ort.InferenceSession(args["Path_model"], providers=['CUDAExecutionProvider'])
        input_name = sess.get_inputs()[0].name
        input_size = sess.get_inputs()[0].shape[-2:]
    elif args["Model_type"] == "caffe":
        caffe.set_mode_gpu()
        caffe.set_device(0)
        net = caffe.Net(args["Path_model"].replace(".caffemodel", ".prototxt"), args["Path_model"], caffe.TEST)
    else:
        print("目前仅支持onnx和caffemodel推理！\n")

    result = dict()
    p_bar = tqdm(path_imgs, ncols=100)
    p_bar.set_description(f'{osp.split(args["Path_model"])[-1]} Processing')
    for path_img in path_imgs: 
        p_bar.update()
        if osp.splitext(path_img)[-1] != ".jpg": continue
        image = cv2.imread(path_img)
        H, W = image.shape[:2]
        # 数据预处理
        img, scale_factor, padding_list = preprocessor(image, input_size)
        # 推理
        if args["Model_type"] == "onnx":
            features = sess.run(None, {input_name: img})
        elif args["Model_type"] == "caffe":
            features = net.forward(data=img)

        # 后处理
        decoder_outputs = decoder(
            features,
            args["Score_thr"],
            num_labels=num_classes,
            anchors=args["Anchors"])

        if len(decoder_outputs[0]) == 0: continue
        bboxes, scores, labels = non_max_suppression(
            *decoder_outputs, 
            args["Score_thr"],
            args["Box_thr"],)

        path_img, pre_label = imshow_det(
            path_img, bboxes, labels,
            save_path=out_dir,
            classes=classes,  
            is_show=is_show, 
            exist_child=exist_child,
            scale_factor=scale_factor,
            padding_list=padding_list)

        result[path_img] = pre_label
    p_bar.close()

    return result

if __name__ == '__main__':
    args = {
        "Path_model": "./model.onnx", 
        "Model_type": "yolov5",
        "Num_classes": 3,
        "Score_thr": 0.3,
        "Box_thr": 0.65,     
        "Anchors": [[(9, 3), (6, 16), (26, 8)], [(15, 40), (32, 73), (63, 130)], [(91, 99), (190, 182),(339, 276)]],      
        "Class_show": {
                "classes": ['Person', 'Plate', 'Car'], 
                "is_show": [1, 1, 1],
                "exist_child": [0, 0, 0]
        }
    }
    inference("./data", args, out_dir='./show')
