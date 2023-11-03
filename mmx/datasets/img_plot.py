import os, copy, cv2, mmcv
import numpy as np
from numpy import random
from mmx.registry import TRANSFORMS


@TRANSFORMS.register_module()
class ClsImgPlot(object):
    def __init__(self, img_save_path='work_dirs/', save_img_num=4, thickness=1):
        self.img_aug_id = 0
        self.img_save_path = img_save_path
        self.save_img_num = save_img_num
        self.thickness = thickness
       
    def __call__(self, results):
        if self.img_aug_id < self.save_img_num: 
            try: os.makedirs(self.img_save_path)
            except: pass
            filename = os.path.join(self.img_save_path, ('img_augment%g.jpg' % self.img_aug_id))
            self.img_aug_id += 1
            img = copy.deepcopy(results['img'])
            label = results['gt_label']
            text = f"Label: {label}"
            text_position = [img.shape[1]//10, img.shape[0]//10]
            img = cv2.putText(img, text, text_position, cv2.FONT_HERSHEY_SIMPLEX, min(img.shape[1], img.shape[0])/400, (0, 0, 255), self.thickness)
            cv2.imwrite(filename, img)
        return results
    
    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'img_save_path={self.img_save_path}, '
        repr_str += f'save_img_num={self.save_img_num}, '
        repr_str += f'thickness={self.thickness}, '
        return repr_st
    

@TRANSFORMS.register_module()
class PolyImgPlot(object):
    """visualize the poly-format img after augmentation.
    Args:
        img_save_path (str): where to save the visualized img.
    """
    def __init__(self, img_save_path='work_dirs/', save_img_num=4, thickness=2, format='rect', colors=[[random.randint(0, 255) for _ in range(3)] for _ in range(20)]):
        self.img_aug_id = 0
        self.img_save_path = img_save_path
        self.save_img_num = save_img_num
        self.colors = colors
        self.thickness = thickness
        assert format in ['rect', 'poly'], 'Error: The format argument must be rect or poly!'
        self.format = format

    def __call__(self, results):       
        if self.img_aug_id < self.save_img_num:
            try: os.makedirs(self.img_save_path)
            except: pass
            filename = os.path.join(self.img_save_path, ('img_augment%g.jpg' % self.img_aug_id))
            self.img_aug_id += 1
            img = copy.deepcopy(results['img'])       # img(h, w, 3) 未归一化
            polys = results['gt_bboxes'].tensor
            labels = results['gt_bboxes_labels']
            if self.format == 'poly':
                # visulize the oriented boxes
                for i, bbox in enumerate(polys):   
                    cls_index = labels[i]
                    box_list = np.array([(bbox[0], bbox[1]), (bbox[2], bbox[3]), (bbox[4], bbox[5]), (bbox[6], bbox[7])], np.int32)
                    cv2.drawContours(image=img, contours=[box_list], contourIdx=-1, color=self.colors[int(cls_index)], thickness=self.thickness)
            else:
                # visulize the oriented boxes
                for i, bbox in enumerate(polys): 
                    cls_index = labels[i]
                    cv2.rectangle(img=img, pt1=[int(bbox[0]), int(bbox[1])], pt2=[int(bbox[2]), int(bbox[3])], color=self.colors[int(cls_index)], thickness=self.thickness, lineType=4)
            cv2.imwrite(filename, img)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'img_save_path={self.img_save_path}, '
        repr_str += f'save_img_num={self.save_img_num}, '
        repr_str += f'colors={self.colors}, '
        repr_str += f'thickness={self.thickness}, '
        repr_str += f'format={self.format})'
        return repr_st

@TRANSFORMS.register_module()
class MaskImgPlot(object):
    """visualize the img after augmentation.
    Args:
        img_save_path (str): where to save the visualized img.
    """
    def __init__(self, img_save_path='work_dirs/', save_img_num=4, palette=None, opacity=0.5):
        self.img_aug_id = 0
        self.img_save_path = img_save_path
        self.save_img_num = save_img_num
        self.palette = palette
        self.opacity = opacity

    def __call__(self, results):       
        if self.img_aug_id < self.save_img_num: 
            try: os.makedirs(self.img_save_path)
            except: pass
            filename = os.path.join(self.img_save_path, ('img_augment%g.jpg' % self.img_aug_id))
            self.img_aug_id += 1
            img = copy.deepcopy(results['img'])
            mask = results['gt_seg_map']
            assert 0 < self.opacity <= 1.0
            color_seg = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
            for label, color in enumerate(self.palette):
                color_seg[mask == label, :] = color
            # convert to BGR
            color_seg = color_seg[..., ::-1]
            img = img * (1 - self.opacity) + color_seg * self.opacity
            img = img.astype(np.uint8)
            mmcv.imwrite(img, filename)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'img_save_path={self.img_save_path}, '
        repr_str += f'save_img_num={self.save_img_num}, '
        repr_str += f'palette={self.palette}, '
        repr_str += f'opacity={self.opacity}, '
        return repr_st
