import random
import os.path as osp

import mmengine
import mmengine.fileio as fileio
from mmx.registry import DATASETS
from mmseg.datasets import BaseSegDataset

# 设置随机种子为0
random.seed(0)

@DATASETS.register_module()
class SegCustomDataset(BaseSegDataset):
    """SegCustomDataset."""

    def __init__(self,
                 auto_sample=False,
                 sample_ratio=0.8,
                 separator=",",
                 ann_file="",
                 img_suffix=".jpg",
                 seg_map_suffix=".png",
                 **kwargs):
        self.separator = separator
        self.auto_sample = auto_sample
        self.sample_ratio = sample_ratio
        super().__init__(img_suffix=img_suffix,
                         seg_map_suffix=seg_map_suffix,
                         ann_file=ann_file,
                         **kwargs)

    def load_data_list(self):
        """Load annotation from directory or annotation file.

        Returns:
            list[dict]: All data info of dataset.
        """
        data_list = []
        img_dir = self.data_prefix.get("img_path", None)
        ann_dir = self.data_prefix.get("seg_map_path", None)
        if osp.isfile(self.ann_file):
            # lines = mmengine.list_from_file(self.ann_file, file_client_args=self.file_client_args)
            lines = mmengine.list_from_file(self.ann_file)
            for line in lines:
                try:
                    img_name, seg_map = line.strip().split(self.separator)
                except:
                    print(line)
                data_info = dict(img_path=osp.join(img_dir, img_name))
                data_info["seg_map_path"] = osp.join(ann_dir, seg_map)
                data_info["label_map"] = self.label_map
                data_info["reduce_zero_label"] = self.reduce_zero_label
                data_info["seg_fields"] = []
                data_list.append(data_info)
        else:
            for img in fileio.list_dir_or_file(dir_path=img_dir,
                                               list_dir=False,
                                               suffix=self.img_suffix,
                                               recursive=True):
                data_info = dict(img_path=osp.join(img_dir, img))
                if ann_dir is not None:
                    seg_map = img.replace(self.img_suffix, self.seg_map_suffix)
                    data_info["seg_map_path"] = osp.join(ann_dir, seg_map)
                data_info["label_map"] = self.label_map
                data_info["reduce_zero_label"] = self.reduce_zero_label
                data_info["seg_fields"] = []
                data_list.append(data_info)
            data_list = sorted(data_list, key=lambda x: x["img_path"])
        if self.auto_sample:
            random.shuffle(data_list)
            if self.sample_ratio > 0.5:
                sample_num = int(len(data_list) * self.sample_ratio)
                return data_list[:sample_num]
            else:
                sample_num = int(len(data_list) * (1 - self.sample_ratio))
                return data_list[sample_num:]
        return data_list