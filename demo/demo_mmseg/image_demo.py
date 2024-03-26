# Copyright (c) OpenMMLab. All rights reserved.
from argparse import ArgumentParser
import sys, ast, warnings
import mmcv
import cv2
import torch
import numpy as np
import os.path as osp
from mmcv.transforms import Compose
from collections import defaultdict
import mmengine.fileio as fileio
from mmengine.config import Config
from mmengine.model import revert_sync_batchnorm
from mmengine.utils import mkdir_or_exist, scandir, ProgressBar
from mmseg.apis import inference_model
sys.path.append(osp.dirname(osp.dirname(osp.dirname(osp.realpath(__file__)))))
from mmx.apis import init_model
warnings.filterwarnings("ignore")
import torch.multiprocessing as mp

def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        'imgs', 
        help='Path to the image files')
    parser.add_argument(
        'config', 
        help='Path to the configuration file for the model')
    parser.add_argument(
        'checkpoint', 
        help='Path to the checkpoint file of the model.')
    parser.add_argument(
        'out', 
        help='Path to the output directory where processed images will be saved')
    parser.add_argument(
        '--GT', 
        default=None,
        help='Optional path to the ground truth label file')
    parser.add_argument(
        '--roi',
        type=ast.literal_eval, 
        default=None,
        help='Optional region of interest in the image as a tuple (x, y, x, y')
    parser.add_argument(
        '--weights',
        type=ast.literal_eval,
        default=None,
        help='Optional weights for different classes in the form of a list')
    parser.add_argument(
        '--num-crop',
        type=int, 
        default=4,
        help='Number of crops to be created from the image')
    parser.add_argument(
        '--device', 
        default='cuda:0', 
        help='Device used for inference')
    parser.add_argument(
        '--opacity',
        type=float,
        default=0.6,
        help='Opacity of painted segmentation map. In (0, 1] range.')
    parser.add_argument(
        '--probs', 
        type=ast.literal_eval, 
        default=[0., 1.],
        help='List of probabilities for data augmentation')
    parser.add_argument(
        '--directions', 
        type=ast.literal_eval, 
        default=['horizontal'],
        help='List of directions for data augmentation')
    parser.add_argument(
        '--thr',
        type=float,
        default=0.8)
    parser.add_argument(
        '--nproc',
        type=int,
        default=16)
    args = parser.parse_args()
    return args

# 结果可视化
def imshow_semantic(img, seg, mask, palette, save_path=None, opacity=0.5, rect=[]):
    mkdir_or_exist(osp.dirname(save_path))
    if isinstance(img, str): img = mmcv.imread(img)
    H, W = seg.shape
    pred_color_seg = np.zeros((H, W, 3), dtype=np.uint8)
    for label, color in enumerate(palette):
        pred_color_seg[seg == label, :] = color
    pred_color_seg = pred_color_seg[..., ::-1]
    pred_img_data = (img * (1 - opacity) + pred_color_seg * opacity).astype(np.uint8)
    # 异常区域添加矩形框显示
    for i in rect:
        cv2.rectangle(pred_img_data, (int(i[0]), int(i[1])), (int(i[2]), int(i[3])), (0, 0, 255), 2)

    # 左边显示原始图片预测结果，右边有Gt则显示Gt，反之显示原始图像
    if mask is not None:
        gt_color_seg = np.zeros((H, W, 3), dtype=np.uint8)
        for label, color in enumerate(palette):
            gt_color_seg[mask == label, :] = color
        gt_color_seg = gt_color_seg[..., ::-1]
        gt_img_data = (img * (1 - opacity) + gt_color_seg * opacity).astype(np.uint8)
    else:
        gt_img_data = img

    drawn_img = np.concatenate((pred_img_data, np.ones((H, W//50, 3), dtype=np.uint8)*255, gt_img_data), axis=1)

    mmcv.imwrite(drawn_img, save_path)

# 计算各个切片区域的置信度矩阵
def process_slice_Acc(i, j, h, w, gt, result_seg_pred, num_classes, weights):
    gt_crop = gt[0, h[i]:h[i+1], w[j]:w[j+1]]
    result_crop = result_seg_pred[0, h[i]:h[i+1], w[j]:w[j+1]]
    inds = (num_classes * gt_crop + result_crop).flatten()
    mat = torch.bincount(inds, minlength=num_classes**2).view(num_classes, num_classes).float()
    per_label_sums = mat.sum(axis=1)
    idx = (per_label_sums > 0) & torch.tensor(weights, dtype=torch.bool)
    per_label_sums[per_label_sums==0] = 1 
    mat /= per_label_sums[:, None]
    avg_acc = mat.diag()[idx].mean()
    return avg_acc.item(), w[j], h[i], w[j+1], h[i+1]

def process_slice_IoU(i, j, h, w, gt, result_seg_pred, num_classes, weights):
    gt_crop = gt[0, h[i]:h[i+1], w[j]:w[j+1]]
    result_crop = result_seg_pred[0, h[i]:h[i+1], w[j]:w[j+1]]
    inds = (num_classes * gt_crop + result_crop).flatten()
    mat = torch.bincount(inds, minlength=num_classes**2).view(num_classes, num_classes).float()
    intersection = torch.diag(mat)
    union = mat.sum(axis=1) + mat.sum(axis=0) - intersection
    idx = (union > 0) & torch.tensor(weights, dtype=torch.bool)
    iou = (intersection / union)[idx].mean()
    return iou.item(), w[j], h[i], w[j+1], h[i+1]

# 一致性校验流程：校验同一输入图像信号经过不同强度的扰动后，预测结果的期望是否一致
def consistency_checking(img, mask, save_path, name, results, palette, thr=0.5, opacity=0.5, weights=[], num_crop=3):
    num_classes = len(palette)
    assert len(weights) == num_classes
    start, i_seg_logits = 0, results[0].get().seg_logits.data.cpu()
    C, H, W = i_seg_logits.shape
    # 有Gt时，输入图像信号经过不同强度的扰动的预测结果取均值，作为pre_result
    # 无Gt时，原始输入图像信号的预测结果作为Gt，其余经过不同强度的扰动的预测结果取均值，作为pre_result
    if mask is None:
        gt, start = results[0].get().pred_sem_seg.data.cpu(), 1
    else: 
        gt = torch.tensor(mask).unsqueeze(0)

    result_seg_logits = torch.zeros(np.shape(i_seg_logits))
    for r in results[start:]:
        result_seg_logits += r.get().seg_logits.data.cpu()
    result_seg_logits =  result_seg_logits/(len(results)-start)
    
    if C > 1: 
        result_seg_pred = result_seg_logits.argmax(dim=0, keepdim=True)
    else:
        result_seg_logits = result_seg_logits.sigmoid()
        result_seg_pred = (result_seg_logits > 0.5).to(result_seg_logits)
    # 为了减少耗时，无法进行像素级校验，因此只能计算各类别像素整体的置信度是否一致，校验精度较差
    # 为了提高校验精度，将图片裁剪为多个切片，再逐一校验
    h = np.linspace(0, H, num_crop+1, dtype=torch.int)
    w = np.linspace(0, W, num_crop+1, dtype=torch.int)
    with mp.Pool(mp.cpu_count()) as pool:
        conf = pool.starmap(process_slice_Acc, [(i, j, h, w, gt, result_seg_pred, num_classes, weights) for i in range(num_crop) for j in range(num_crop)])
    
    conf = torch.tensor(conf)
    conf = conf.view(num_crop, num_crop, 5)        

    low_conf_idx = conf[:,:,0] < thr
    if low_conf_idx.any(): floder = 'inconsistent'
    else: floder = 'consistent'

    imshow_semantic(
        img, 
        result_seg_pred[0],
        None if mask is None else gt[0],
        palette,
        save_path=osp.join(osp.join(save_path, floder), name),
        opacity=opacity,
        rect=conf[low_conf_idx][:, 1:])

def get_data(img_path, gt, roi):
    image = mmcv.imread(img_path)
    # 支持有/无Gt
    if gt: 
        img_bytes = fileio.get(osp.join(gt, osp.split(img_path)[-1].replace('.jpg', '.png')))
        mask = mmcv.imfrombytes(img_bytes, flag='unchanged').squeeze().astype(np.uint8)
    else: mask = None

    # 支持设定规则区域
    if isinstance(roi, tuple):
        crop_x1, crop_y1, crop_x2, crop_y2 = roi
        image = image[crop_y1:crop_y2, crop_x1:crop_x2]
        if not mask: mask = mask[crop_y1:crop_y2, crop_x1:crop_x2]
    elif isinstance(roi, dict):
        key = '_'.join(osp.split(img_path)[-1].split('_')[:-1])
        try:
            crop_x1, crop_y1, crop_x2, crop_y2 = roi[key]
            image = image[crop_y1:crop_y2, crop_x1:crop_x2]
            if not mask: mask = mask[crop_y1:crop_y2, crop_x1:crop_x2]
        except: pass
    elif roi == None: pass
    else: print("ROI设置有误！")
    return image, mask

def inference_detector_custom(args, img, img_scale, prob, direction):
    
    model = init_model(args.config, args.checkpoint, device=args.device)
    if args.device == 'cpu':
        model = revert_sync_batchnorm(model)

    # 通过尺度变换、翻转操作给同一图像施加不同程度的输入扰动
    pipeline = [
        dict(type='LoadImageFromFile'),
        dict(
            type='Resize', 
            scale=img_scale, 
            keep_ratio=False),
        dict(
            type='RandomFlip', 
            direction=direction,
            prob=prob),
        dict(type='PackSegInputs')
    ]

    data = defaultdict()
    if model.data_preprocessor.device.type == 'cpu':
        for m in model.modules():
            assert not isinstance(m, RoIPool)

    if isinstance(img, np.ndarray):
        data_ = dict(img=img)
        pipeline[0]['type'] = 'LoadImageFromNDArray'
    else:
        data_ = dict(img_path=img)
    
    test_pipeline = Compose(pipeline)
    data_ = test_pipeline(data_)
    data['inputs'] = [data_['inputs']]
    data['data_samples'] = [data_['data_samples']]
    with torch.no_grad():
        result = model.test_step(data)[0]
    return result

def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)
    img_scale = cfg.img_scale
    palette = cfg.metainfo.palette
    palette = np.array(palette)
    imglist = list(scandir(args.imgs, ('.jpg', '.png'), recursive=True))
    
    ctx = torch.multiprocessing.get_context("spawn")
    num_tasks = len(imglist)
    pool = ctx.Pool(args.nproc if num_tasks > args.nproc else num_tasks)

    prog_bar = ProgressBar(num_tasks, num_tasks, True, file=sys.stdout)
    for img in imglist:
        img_path = osp.join(args.imgs, img)
        image, mask = get_data(img_path, args.GT, args.roi)

        results = []
        for prob in args.probs:
            for direction in args.directions:
                result = pool.apply_async(
                    inference_detector_custom, 
                    args=(args, image, img_scale, prob, direction))
                results.append(result)

        # 一致性校验
        consistency_checking(
            image,
            mask,
            args.out, 
            img,
            results, 
            palette, 
            args.thr,
            args.opacity,
            np.repeat(1, len(palette)) if not args.weights else args.weights,args.num_crop)
        
        prog_bar.update()

    pool.close()
    pool.join()
    prog_bar.file.write('\n')

if __name__ == '__main__':
    main()
