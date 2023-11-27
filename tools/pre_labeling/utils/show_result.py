# -*- coding: utf-8 -*-
import os, math
import cv2
import os.path as osp
import numpy as np

colors = [np.random.randint(0, 255, 3) for _ in range(20)]

def imshow_semantic(img, seg, label=None, palette=colors, save_path=None, opacity = 0.8, rect=None):
    """分割结果可视化
    Args:
        img (str | ndarray): 图片路径 or 图片数据
        seg (ndarray): 原图坐标系下的分割掩码结果
        mask (ndarray): 掩码GT
        palette (list): 各类别颜色
        save_path (str): 可视化结果存储路径
        opacity (float): 透明度 0 ~ 1
        rect (list): 目标框信息
    """
    if isinstance(img, str): 
        image = cv2.imread(img)
    else:
        image = img

    H, W, _ = image.shape

    index_list = np.unique(seg)
    mask = np.zeros((H, W)).astype(np.uint8)
    color_seg = np.zeros([H, W, 3]).astype(np.uint8)
    for index in index_list:
        if index == 0: continue
        mask[seg == index] = index
        color_seg[seg == index] = palette[index]
    image_seg = image * (1-opacity) + color_seg[..., ::-1] * opacity

    # 保存预测掩码结果
    if isinstance(img, str): 
        root = osp.join(osp.dirname(osp.dirname(img)), 'mask')
        if not osp.exists(root): os.makedirs(root)
        path_mask = osp.join(root, osp.split(img)[-1].replace(".jpg", ".png"))
        cv2.imwrite(path_mask, mask)

    # 异常区域添加矩形框显示
    if rect:
        for i in rect:
            cv2.rectangle(image_seg, (int(i[0]), int(i[1])), (int(i[2]), int(i[3])), (0, 0, 255), 2)
    
    # 左边显示原始图片预测结果，右边有Gt则显示Gt，反之显示原始图像
    if label is not None:
        mask_gt = np.zeros((H, W)).astype(np.uint8)
        color_seg_gt = np.zeros([H, W, 3]).astype(np.uint8)
        for index in index_list:
            if index == 0: continue
            mask_gt[label == index] = index
            color_seg_gt[label == index] = palette[index]
        image_seg_gt = image * (1-opacity) + color_seg_gt[..., ::-1] * opacity
    else:
        image_seg_gt = image
    image_show = np.concatenate((image_seg, np.ones((H, W//50, 3), dtype=np.uint8)*255, image_seg_gt), axis=1)

    if save_path: pass
    elif isinstance(img, str):
        root = osp.join(osp.dirname(osp.dirname(img)), 'show')
        if not osp.exists(root): os.makedirs(root)
        save_path = osp.join(root, osp.split(img)[-1])
    else:
        print(f"输入数据为{type(img)}，请指定 save_path！")
    
    cv2.imwrite(save_path, image_show)

def imshow_det(img, bboxes, labels, palette=colors, save_path=None, font_scale=0.5, thickness=4, **kargs):
    """检测结果可视化
    Args:
        img (str | ndarray): 图片路径 or 图片数据
        bboxes (list): 检测结果。如果检测结果不是在原图坐标系下，还需输入缩放系数scale_factor和padding_list
        labels (list): bboxes对应的预测类别
        palette (list): 各类别颜色
        save_path (str): 可视化结果存储路径
        font_scale (float): 缩放系数 0 ~ 1
        thickness (int): 线条粗细
    Returns: 
    """
    # 支持仅展示部分目标
    is_show = kargs.get("is_show", [True]*80)
    classes = kargs.get("classes", None)
    
    if isinstance(img, str): 
        image = cv2.imread(img)
    else:
        image = img

    H, W = image.shape[:2]
    ratio_h, ratio_w = kargs.get("scale_factor", [1, 1])
    padding_list = kargs.get("padding_list", [0]*4)
    pre_label = []
    for box, label in zip(bboxes, labels):
        if not is_show[label]: continue
        if classes: 
            cls = classes[label]
        else:
            cls = str(label)
        x0, y0, x1, y1 = box - np.array(padding_list[2], padding_list[0], padding_list[2], padding_list[0])
        x0 = math.floor(min(max(x0 / ratio_w, 1), W - 1))
        y0 = math.floor(min(max(y0 / ratio_h, 1), H - 1))
        x1 = math.ceil(min(max(x1 / ratio_w, 1), W - 1))
        y1 = math.ceil(min(max(y1 / ratio_h, 1), H - 1))
        pre_label.append([x0, y0, x1, y1, cls])
        if save_path:
            if min(y1-y0, x1-x0) < 100: thickness = 2
            ((text_width, text_height), _) = cv2.getTextSize(
                cls, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 2)
            cv2.rectangle(image, (x0, y0), (x1, y1), tuple([int(i) for i in palette[label]]), thickness)
            if (x1 - x0) > text_width:
                cv2.rectangle(image, (x0, y0), (x0 + text_width, y0 + int(1.3 * text_height)), (0, 0, 0), -1)
                cv2.putText(image, cls, (x0, y0 + int(text_height)), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), lineType=cv2.LINE_AA)
                    
        if save_path and isinstance(img, str):
            if not osp.exists(save_path): os.makedirs(save_path)
            cv2.imwrite(osp.join(save_path, osp.split(img)[-1]), image)
    return pre_label