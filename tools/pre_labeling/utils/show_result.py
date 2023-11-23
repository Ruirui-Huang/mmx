import os
import cv2
import os.path as osp
import numpy as np

colors = [np.random.randint(0, 255, 3) for _ in range(20)]

def imshow_semantic(img, seg, label=None, palette=colors, save_path=None, opacity = 0.8, rect=None):
        """分割结果可视化
        Args:
            img: 图片路径 or 图片数据
            seg: 原图坐标系下的分割掩码结果
            mask: 掩码GT
            palette: 各类别颜色
            save_path: 可视化结果存储路径
            opacity: 透明度
            rect: 目标框信息
        Returns: 
        """
        if isinstance(img, str): img = cv2.imread(img)
        H, W, _ = img.shape

        index_list = np.unique(seg)
        mask = np.zeros((H, W)).astype(np.uint8)
        color_seg = np.zeros([H, W, 3]).astype(np.uint8)
        for index in index_list:
            mask[seg == index] = index
            color_seg[seg == index] = palette[index]

        img_seg = img * (1-opacity) + color_seg[..., ::-1] * opacity

        if isinstance(img, str): 
            root = osp.join(osp.dirname(img), 'mask')
            if not osp.exists(root): os.makedirs(root)
            path_mask = osp.join(root, osp.splitext(img)[-1].replace(".jpg", ".png"))
            cv2.imwrite(path_mask, mask)
        # 异常区域添加矩形框显示
        if rect:
             for i in rect:
                cv2.rectangle(img_seg, (int(i[0]), int(i[1])), (int(i[2]), int(i[3])), (0, 0, 255), 2)
        
        # 左边显示原始图片预测结果，右边有Gt则显示Gt，反之显示原始图像
        if label is not None:
            mask_gt = np.zeros((H, W)).astype(np.uint8)
            color_seg_gt = np.zeros([H, W, 3]).astype(np.uint8)
            for index in index_list:
                mask_gt[label == index] = index
                color_seg_gt[label == index] = palette[index]

            img_seg_gt = img * (1-opacity) + color_seg_gt[..., ::-1] * opacity
        else:
            img_seg_gt = img

        img = np.concatenate((img_seg, np.ones((H, W//50, 3), dtype=np.uint8)*255, img_seg_gt), axis=1)
        if save_path: pass
        elif isinstance(img, str):
            root = osp.join(osp.dirname(img), 'show')
            if not osp.exists(root): os.makedirs(root)
            path_mask = osp.join(root, osp.splitext(img)[-1])
        cv2.imwrite(save_path, img)