import os
import cv2
import os.path as osp
import numpy as np

colors = [np.random.randint(0, 255, 3) for _ in range(20)]

def ShowMask(mask, path_img, opacity = 0.8):
        """分割结果可视化
        Args:
            mask: 原图坐标系下的分割掩码结果
            path_img: 图片路径
        Returns: 
        """
        image = cv2.imread(path_img)
        H, W, _ = image.shape

        index_list = np.unique(mask)
        mask = np.zeros((H, W)).astype(np.uint8)
        color_seg = np.zeros([H, W, 3]).astype(np.uint8)
        for index in index_list:
            mask[mask == index] = index
            color_seg[mask == index] = colors[index]
        path_mask = osp.splitext(path_img.replace('/data/', '/mask/'))[0] +  ".png"
        filedir = osp.dirname(path_mask)
        if not osp.exists(filedir): os.makedirs(filedir)
        cv2.imwrite(path_mask, mask)

        image = image * (1-opacity) + color_seg[..., ::-1] * opacity
        img = np.concatenate((image, np.ones((H, W//50, 3), dtype=np.uint8)*255, image), axis=1)
        path_show = path_img.replace('/data/', '/show/')
        filedir = osp.dirname(path_show)
        if not osp.exists(filedir): os.makedirs(filedir)
        cv2.imwrite(path_show, img)