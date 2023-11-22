import cv2
import numpy as np
import math
import random
import os.path as osp
import onnxruntime
from pathlib import Path
from config import ModelType
from tqdm import tqdm
from argparse import ArgumentParser
from preprocess import Preprocess
from numpy_coder import Decoder
from cv2_nms import non_max_suppression

def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        'imgs', help='Images path.')
    parser.add_argument(
        'onnx', type=str, help='Onnx file')
    parser.add_argument(
        '--type', type=str, help='Model type')
    parser.add_argument(
        '--class_show', type=dict, help="Including 'class_name: is_show' ")
    parser.add_argument(
        '--anchors', nargs='+', type=int, help='Anchors')
    parser.add_argument(
        '--path-text', default=None, type=str, help='Save .txt file')
    parser.add_argument(
        '--out-dir', default='./output', type=str, help='Path to output file')
    parser.add_argument(
        '--score-thr', type=float, default=0.3, help='Bbox score threshold')
    parser.add_argument(
        '--iou-thr', type=float, default=0.7, help='Bbox iou threshold')
    args = parser.parse_args()
    return args

IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp')

def path_to_list(path: str):
    path = Path(path)
    if path.is_file() and path.suffix in IMG_EXTENSIONS:
        res_list = [str(path.absolute())]
    elif path.is_dir():
        res_list = [
            str(p.absolute()) for p in path.iterdir()
            if p.suffix in IMG_EXTENSIONS
        ]
    else:
        raise RuntimeError
    return res_list

def main():
    args = parse_args()
    assert args.class_show, print("class_show为空！")
    model_type = ModelType(args.type.lower())
    num_labels=len(args.class_show)
    colors=[[random.randint(0, 255) for _ in range(3)] for _ in range(num_labels)]
    preprocessor = Preprocess()
    session = onnxruntime.InferenceSession(
            args.onnx, 
            providers=['CUDAExecutionProvider'])
    decoder = Decoder(model_only=True)
    input_name = session.get_inputs()[0].name
    img_size = session.get_inputs()[0].shape[-2:]
    if args.path_txt: fw = open(args.path_txt, 'w+')
    files = path_to_list(args.imgs)
    for file in tqdm(files):
        image = cv2.imread(file)
        image_h, image_w = image.shape[:2]
        img, (ratio_w, ratio_h) = preprocessor(image, img_size)
        features = session.run(None, {{input_name: img}})
        decoder_outputs = decoder(
            features,
            args.score_thr,
            num_labels=num_labels,
            anchors=args.anchors)
        nmsd_boxes, nmsd_scores, nmsd_labels = non_max_suppression(
            *decoder_outputs, args.score_thr, args.iou_thr)
        pre_label = []
        for box, score, label in zip(nmsd_boxes, nmsd_scores, nmsd_labels):
            for cls, show in args.class_show:
                if not show: continue
                x0, y0, x1, y1 = box
                x0 = math.floor(min(max(x0 / ratio_w, 1), image_w - 1))
                y0 = math.floor(min(max(y0 / ratio_h, 1), image_h - 1))
                x1 = math.ceil(min(max(x1 / ratio_w, 1), image_w - 1))
                y1 = math.ceil(min(max(y1 / ratio_h, 1), image_h - 1))

                pre_label.append(f'{x0},{y0},{x1},{y1},{cls}')

                if args.out_dir:
                    cv2.rectangle(image, (x0, y0), (x1, y1), colors[label], 2)
                    cv2.putText(image, f'{cls}: {score:.2f}',
                                (x0, y0 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                (0, 255, 255), 2)
        if args.out_dir:
            cv2.imwrite(osp.join(args.out_dir, osp.split(file)[-1]), image)
        
        if args.path_txt:  
            fw.write(file + ' ' + ' '.join(pre_label) + '\n') 

    if args.path_txt: 
        fw.close()

if __name__ == '__main__':
    main()

