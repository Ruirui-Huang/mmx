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

def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        'imgs', help='Images path.')
    parser.add_argument(
        'onnx', type=str, help='Onnx file')
    parser.add_argument(
        '--model-type', type=str, help='Model type', default='yolov5')
    parser.add_argument(
        '--class-show', type=dict, help="Including 'class_name: is_show' ", default={"classes": ['Person', 'Plate', 'Car'], "is_show": [True, True, True]})
    parser.add_argument(
        '--anchors', nargs='+', type=int, help='Anchors', default=[[(9, 3), (6, 16), (26, 8)], [(15, 40), (32, 73), (63, 130)], [(91, 99), (190, 182), (339, 276)]])
    parser.add_argument(
        '--score-thr', type=float, default=0.3, help='Bbox score threshold')
    parser.add_argument(
        '--iou-thr', type=float, default=0.7, help='Bbox iou threshold')
    parser.add_argument(
        '--out-dir', default='./show', type=str, help='Path to output file')
    args = parser.parse_args()
    return args


def onnx_od(args):
    """ONNX前向，用于目标检测推理
    Args:
    Returns: 推理结果，及渲染之后的图片
    """
    assert args.class_show, print("class_show为空！")
    classes = args.class_show["classes"]
    is_show = args.class_show["is_show"]
    num_classes = len(classes)
    assert num_classes == len(is_show), print("num_classes和is_show长度不一致！")
    colors=[[random.randint(0, 255) for _ in range(3)] for _ in range(num_classes)]
    preprocessor = Preprocess()
    # 加载引擎
    sess = ort.InferenceSession(args.onnx, providers=['CUDAExecutionProvider'])
    decoder = Decoder(model_type=args.model_type, model_only=True)
    input_name = sess.get_inputs()[0].name
    input_size = sess.get_inputs()[0].shape[-2:]
    files = path_to_list(args.imgs)
    result = dict()
    for file in tqdm(files): 
        image = cv2.imread(file)
        H, W = image.shape[:2]
        # 数据预处理
        img, (ratio_w, ratio_h) = preprocessor(image, input_size)
        # 推理
        features = sess.run(None, {input_name: img})
        # 后处理
        decoder_outputs = decoder(
            features,
            args.score_thr,
            num_labels=num_classes,
            anchors=args.anchors)

        if len(decoder_outputs[0]) == 0: continue
        
        bboxes, scores, labels = non_max_suppression(
            *decoder_outputs, 
            args.score_thr, 
            args.iou_thr)

        pre_label = imshow_det(
            file, bboxes, labels,             save_path=args.out_dir, is_show=is_show, 
            classes=classes, 
            input_size=input_size)

        result[file] = pre_label

    return result

if __name__ == '__main__':
    args = parse_args()
    onnx_od(args)