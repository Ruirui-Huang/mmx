# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
import torch
from argparse import ArgumentParser

import mmcv
from mmcv.transforms import Compose
from mmcv.ops import batched_nms
from mmdet.apis import inference_detector, init_detector
from mmdet.structures.bbox import bbox_flip
from mmengine.logging import print_log
from mmengine.config import Config
from mmengine.utils import ProgressBar, mkdir_or_exist
from mmengine.structures import InstanceData
from mmyolo.registry import VISUALIZERS
from mmyolo.utils import register_all_modules, switch_to_deploy
from mmyolo.utils.misc import get_file_list
from ensemble_boxes import weighted_boxes_fusion as WBF
import ast

def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        'img', help='Image path, include image file, dir and URL.')
    parser.add_argument('config', help='Config file')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument('--path-txt', default=None, help='Save .txt file')
    parser.add_argument('--save-result', default=False)
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument('--score-thr', type=ast.literal_eval, default=0.3)
    parser.add_argument('--scale-factors', type=ast.literal_eval, default=[1.0])
    parser.add_argument('--probs', type=ast.literal_eval, default=[0., 1.])
    parser.add_argument('--directions', type=ast.literal_eval, default=['horizontal'])
    args = parser.parse_args()
    return args


def inference_detector_custom(model, img, scale_factors, probs, directions, iou_thr):
    bboxes_list = []
    scores_list = []
    labels_list = []
    img_scale = list(model.cfg.img_scale)
    for scale_factor in scale_factors:
        img_scale = tuple([int(i*scale_factor) for i in img_scale])
        for prob in probs:
            for direction in directions:
                pipeline = [
                    dict(
                        type='LoadImageFromFile', 
                        file_client_args=dict(backend='disk')),
                    dict(
                        type='LoadAnnotations', 
                        with_bbox=True, 
                        _scope_='mmdet'),
                    dict(
                        type='YOLOv5KeepRatioResize', 
                        scale=img_scale),
                    dict(
                        type='mmdet.Pad', 
                        size_divisor=32, 
                        pad_val=114),
                    dict(
                        type='mmdet.RandomFlip', 
                        direction=direction,
                        prob=prob),
                    dict(
                        type='mmdet.PackDetInputs',
                        meta_keys=('img_path', 'ori_shape', 'img_shape', 'scale_factor', 'flip', 'flip_direction'))]
                    
                test_pipeline = Compose(pipeline)

                if model.data_preprocessor.device.type == 'cpu':
                    for m in model.modules():
                        assert not isinstance(m, RoIPool)

                data_ = dict(img_id=0, img_path=img)
                data_ = test_pipeline(data_)
                data_['inputs'] = [data_['inputs']]
                data_['data_samples'] = [data_['data_samples']]

                with torch.no_grad():
                    result = model.test_step(data_)[0]
                    H, W = result.ori_shape
                    if result.flip:
                        result.pred_instances.bboxes = bbox_flip(
                            bboxes=result.pred_instances.bboxes, img_shape=result.ori_shape, 
                            direction=result.flip_direction)

                bboxes_list.append(result.pred_instances.bboxes/torch.tensor([W, H, W, H]).cuda() if len(result.pred_instances.bboxes) != 0 else result.pred_instances.bboxes)
                scores_list.append(result.pred_instances.scores.cpu().tolist())
                labels_list.append(result.pred_instances.labels.cpu().tolist())
    
    weights = [1]*(len(directions)*len(probs)*len(scale_factors))
    weights[0] = 2
    boxes, scores, labels = WBF(
        bboxes_list, 
        scores_list, 
        labels_list,
        weights=weights,
        iou_thr=iou_thr,
        skip_box_thr=0.0001)

    tmp = InstanceData()
    tmp.bboxes = (torch.from_numpy(boxes)*torch.tensor([W, H, W, H])).cuda()
    tmp.scores = torch.from_numpy(scores).cuda()
    tmp.labels = torch.from_numpy(labels).cuda()
    result.pred_instances = tmp
    return result

def main():
    args = parse_args()
    register_all_modules()

    model = init_detector(args.config, args.checkpoint, device=args.device)
    classes = model.cfg.metainfo.CLASSES

    if isinstance(args.score_thr, float):
        score_thr = args.score_thr
    elif isinstance(args.score_thr, dict):
        try: score_thr = args.score_thr[classes[0]]
        except:
            print("aat -> score_thr 中目标不存在\n")
            return
    else:
        print("score_thr 输入有误！\n")
        return
    
    if args.path_txt: fw = open(args.path_txt, 'w+')
    # init visualizer
    visualizer = VISUALIZERS.build(model.cfg.visualizer)
    visualizer.dataset_meta = model.dataset_meta

    # # get file list
    files, source_type = get_file_list(args.img)

    # # start detector inference
    progress_bar = ProgressBar(len(files))
    for file in files:
        result = inference_detector_custom(model, file, scale_factors=args.scale_factors, probs=args.probs, directions=args.directions, iou_thr=score_thr)
        if args.save_result:
            if osp.splitext(file)[-1] in [".jpg", ".JPG", ".png", ".PNG", ".bmp"]:
                img = mmcv.imread(file)
                img = mmcv.imconvert(img, 'bgr', 'rgb')
                out_dir = osp.join(osp.dirname(args.img), 'result')
                mkdir_or_exist(out_dir)
                filename = osp.split(file)[-1]
                visualizer.add_datasample(
                    filename,
                    img,
                    data_sample=result,
                    draw_gt=False,
                    show=False,
                    wait_time=0,
                    out_file=osp.join(out_dir, filename),
                    pred_score_thr=score_thr)
        
        if args.path_txt:
            scores = result.pred_instances.scores 
            boxes = result.pred_instances.bboxes
            labels = result.pred_instances.labels
            
            new_boxes = torch.empty(1, 4).cuda()
            new_labels = torch.empty(1, 1).cuda()
            for i, j in enumerate(scores):
                if j > score_thr: 
                    new_boxes = torch.cat((new_boxes, boxes[i].reshape(1, 4)), dim=0)
                    new_labels = torch.cat((new_labels, labels[i].reshape(1, 1)), dim=0)
            if len(new_labels) == 1: continue
            new_boxes = torch.cat((new_boxes[1:], new_labels[1:].reshape(len(new_labels)-1, 1)), 1).cpu().tolist()

            pre_label = ' '.join([",".join([str(max(int(i), 0)) for i in box[:-1]]) + ',' + classes[int(box[-1])] for box in new_boxes])
            fw.write(file + ' ' + pre_label + '\n') 
        progress_bar.update()
    if args.path_txt: fw.close()

if __name__ == '__main__':
    main()