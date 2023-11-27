import cv2
import numpy as np

class Preprocess:
    def __init__(self, fixed_scale=0, color_space="rgb"):
        self.mean = np.array([0, 0, 0], dtype=np.float32).reshape((3, 1, 1))
        self.std = np.array([255, 255, 255], dtype=np.float32).reshape((3, 1, 1))
        self.fixed_scale = fixed_scale
        self.is_rgb = True if color_space == "rgb" else False

    def __call__(self, image, new_size):
        padding_list = []
        height, width = image.shape[:2]
        height_new, width_new = new_size
        ratio_h, ratio_w = height_new / height, width_new / width
        if self.fixed_scale == 0:
            image = cv2.resize(image, (width_new, height_new))
        else:
            no_pad_shape = (int(round(height*ratio_h)), int(round(width * ratio_w)))
            padding_h, padding_w = [height_new - no_pad_shape[0], width_new - no_pad_shape[1]]
    
            if (height, width) != no_pad_shape:
                image = cv2.resize(image, (no_pad_shape[1], no_pad_shape[0]))

            if self.fixed_scale == 1:
                top_padding, left_padding = int(round(padding_h // 2 - 0.1)), int(round(padding_w // 2 - 0.1))
                bottom_padding = padding_h - top_padding
                right_padding = padding_w - left_padding

            padding_list = [top_padding, bottom_padding, left_padding, right_padding]
            if top_padding != 0 or bottom_padding != 0 or \
                    left_padding != 0 or right_padding != 0:
                image = cv2.copyMakeBorder(
                    image,
                    bottom_padding,
                    right_padding,
                    top_padding,
                    left_padding,
                    cv2.BORDER_CONSTANT,
                    value=(114, 114, 114)
                )
        image = np.ascontiguousarray(image.transpose(2, 0, 1))
        image = image.astype(np.float32)
        image -= self.mean
        image /= self.std
        return image[np.newaxis], (ratio_w, ratio_h), padding_list