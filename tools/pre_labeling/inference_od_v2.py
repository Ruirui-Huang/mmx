# -*- coding: utf-8 -*-
import os, shutil, argparse, json, warnings
warnings.filterwarnings("ignore")
import cv2
import os.path as osp
import pandas as pd
from tqdm import tqdm
from multiprocessing import Pool
from mmengine.config import Config
from utils import NpEncoder, onnx_od, read_cfg, deal_unlabeled_sample, imshow_det

def getArgs():
    parser = argparse.ArgumentParser(description="目标检测预标注！")
    parser.add_argument('--token', type=str, required=True)
    parser.add_argument('--jsonFile', type=str, required=True)
    parser.add_argument('--filesInfo', type=str, required=True)
    return parser

class Prelabeling:
    def __init__(self, args):
        fileInfo = open(args.filesInfo, "r", encoding="utf-8")
        jsonFile = open(args.jsonFile, mode='r', encoding='utf-8')
        self.info = [json.loads(info) for info in fileInfo.readlines()]
        self.inputInfo = json.load(jsonFile)
        jsonFile.close()
        fileInfo.close()
        self.nproc = self.inputInfo["args"]["n_process"]
        self.path_imgs = [info["file"] for info in self.info]

    def onnx_inference(self, onnx_map):
        """ONNX推理
        Args:
        Returns: 
        """
        self.df = pd.DataFrame(columns=['path_img', 'labels'])
        pool = Pool(self.nproc)
        for map in onnx_map:
            pool.apply_async(
                func=onnx_od, 
                args=(map, ), 
                callback=self.callback_merge_labels)
        pool.close()
        pool.join()

    def callback_merge_labels(self, result):
        """预标注结果合并
        Args:
            result: 预标注结果[path_img, labels]
        Returns: 
        """
        for path_img, labels in result.items():
            if not len(labels): print(f"{path_img}中不存在目标！")
            if path_img not in self.df['path_img'].values:
                self.df.loc[len(self.df)] = [path_img, labels]
            else:
                index = self.df['path_img'][self.df['path_img'].values == path_img].index
                self.df.loc[index, 'labels'].append(labels)

    def generate_single_json(self, info_base, info_od1, info_od2):
        """预标注结果写入
        Args:
            info_base: 图片基本信息
            info_od1: 一级od预标结果
            info_od2: 二级od预标结果
        Returns: 
        """
        path_img = info_base["file"]
        path_json = info_base["json"].replace(self.inputInfo["markDataInPath"], self.inputInfo["markDataOutPath"])
        image = cv2.imread(path_img)
        H, W, _ = image.shape

        output_info = {
            "baseId": info_base["baseId"],
            "fileName": osp.basename(path_img),
            "imgSize": [W, H],
            "objects": []
        }
        obj_idx = 1
        for label in info_od1:
            x0, y0, x1, y1, cls = label
            obj_json = {
                    "class": cls,
                    "parent": [],
                    "coord": [[x0, y0], [x1, y1]],
                    "id": self.obj_idx,
                    "shape": "poly",
                    "props": {}
                }
            output_info["objects"].append(obj_json)
            obj_idx += 1

        return path_json, output_info, info_od2
        
    def callback_merge_json(self, path_json, parent, child):
        """一级od和二级od嵌套
        Args:
            path_json: 预标注json路径
            parent: 一级od标注信息
            child: 二级od标注信息
        Returns: 
        """
        fileDir = osp.dirname(path_json)
        if not osp.exists(fileDir): os.makedirs(fileDir)
        with open(path_json, 'w') as json_f:
            json.dump(output_info, json_f, indent=2, cls=NpEncoder)
        json_f.close()
        
    def run(self):
        cfg = Config.fromfile('./config.py')
        PRELABELING_MAP = cfg.get('PRELABELING_MAP', None)
        assert PRELABELING_MAP, print("检查PRELABELING_MAP配置！")
        parent_map, child_map = read_cfg(PRELABELING_MAP)
        # 一级OD
        self.onnx_inference(parent_map)
        df_od1 = self.df
        # 二级OD
        if child_map:
            self.onnx_inference(child_map)
            df_od2 = self.df
        # json嵌套
        pool = Pool(self.nproc)
        for info_base in self.info:
            index = df_od1['path_img'][df_od1['path_img'].values == path_img].index
            info_od1 = df_od1.loc[index, 'labels']
            info_od2 = None
            pool.apply_async(
                func=self.generate_single_json, 
                args=(info_base, info_od1, info_od2, ),
                callback=self.callback_merge_json
            )
        pool.close()
        pool.join()

        # 线下预标开启看效果，以及没有标注结果的情况
        if bool(self.inputInfo["args"]["show_result"]):
            deal_unlabeled_sample(self.path_imgs, self.inputInfo["markDataOutPath"])
            for path_img in self.path_imgs:
                imshow_det(path_img, bboxes, labels, save_path=osp.join(osp.dirname(osp.dirname(path_img)), 'show'))

if __name__ == '__main__':
    parser = getArgs()
    args = parser.parse_args()
    p = Prelabeling(args)
    p.run()
