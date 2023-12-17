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
        label (ndarray): 掩码GT
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

def imshow_det(path_img, bboxes, llabels, palette=colors, save_path=None, font_scale=0.5, thickness=4, **kargs):
    """检测结果可视化
    Args:
        path_img (str): 图片路径
        bboxes (list): 检测结果。如果检测结果不是在原图坐标系下，还需输入缩放系数scale_factor 和 padding_list
        llabels (list): bboxes对应的预测类别
        palette (list): 各类别颜色
        save_path (str): 可视化结果存储路径
        font_scale (float): 缩放系数 0 ~ 1
        thickness (int): 线条粗细
    Returns: 
    """
    classes = kargs.get("classes", None)
    is_show = kargs.get("is_show", [1]*80)
    exist_child = kargs.get("exist_child", [0]*80)
    padding_list = kargs.get("padding_list", [0]*4)
    ratio_h, ratio_w = kargs.get("scale_factor", [1, 1])

    note = None
    root, img = osp.split(path_img)
    image = cv2.imread(path_img)
    H, W = image.shape[:2]

    # 如果是抠图的检测结果，解析图片路径中的父级信息，便于子级坐标转换
    if "crop_imgs" in root:
        img_name_new = img.split("_")
        img_name = "_".join(img_name_new[:-6])
        cls_parent = img_name_new[-6]
        x0_parent = int(img_name_new[-5])
        y0_parent = int(img_name_new[-4])
        x1_parent = int(img_name_new[-3])
        y1_parent = int(img_name_new[-2])
        # 预测结果附带上父级，便于写入json时找父级id
        note = f"{cls_parent}_{x0_parent}_{y0_parent}_{x1_parent}_{y1_parent}"

    pre_label = []
    for box, label in zip(bboxes, llabels):
        if not is_show[label]: continue
        if classes: 
            cls = classes[label]
        else:
            cls = str(label)

        # 按数据预处理方式，将检测结果逆向转换到原图坐标系
        x0, y0, x1, y1 = box - np.array([padding_list[2], padding_list[0], padding_list[2], padding_list[0]])
        x0 = math.floor(min(max(x0 / ratio_w, 1), W - 1))
        y0 = math.floor(min(max(y0 / ratio_h, 1), H - 1))
        x1 = math.ceil(min(max(x1 / ratio_w, 1), W - 1))
        y1 = math.ceil(min(max(y1 / ratio_h, 1), H - 1))
        
        # 如果存在子级，抠图并保存，图片路径包含抠图信息
        if exist_child[label]:
            image_crop = image[y0:y1, x0:x1]
            img_name, _ = osp.splitext(img)
            path_crop_imgs = osp.join(root, "crop_imgs")
            if not osp.exists(path_crop_imgs): os.makedirs(path_crop_imgs)
            img_name_new = img_name + f"_{cls}_{x0}_{y0}_{x1}_{y1}_.jpg"
            cv2.imwrite(osp.join(path_crop_imgs, img_name_new), image_crop)
        
        # 如果是抠图的检测结果，将检测结果转换到原图坐标系
        if "crop_imgs" in root:
            x0 = x0 + x0_parent
            y0 = y0 + y0_parent
            x1 = x1 + x0_parent
            y1 = y1 + y0_parent
        
        pre_label.append([x0, y0, x1, y1, cls, note])

        if save_path:
            if min(y1-y0, x1-x0) < 100: thickness = 2
            ((text_width, text_height), _) = cv2.getTextSize(
                cls, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 2)
            cv2.rectangle(image, (x0, y0), (x1, y1), tuple([int(i) for i in palette[label]]), thickness)
            if (x1 - x0) > text_width:
                cv2.rectangle(image, (x0, y0), (x0 + text_width, y0 + int(1.3 * text_height)), (0, 0, 0), -1)
                cv2.putText(image, cls, (x0, y0 + int(text_height)), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), lineType=cv2.LINE_AA)   
            if not osp.exists(save_path): os.makedirs(save_path)
            cv2.imwrite(osp.join(save_path, img), image)

    path_img = osp.join(osp.dirname(root), img_name + ".jpg")
    
    return path_img, pre_label