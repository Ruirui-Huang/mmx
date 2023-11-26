# -*- coding: utf-8 -*-
import os, argparse, json, warnings
warnings.filterwarnings("ignore")
import cv2
import os.path as osp
import pandas as pd
from multiprocessing import Pool
from mmengine.config import Config
from utils import NpEncoder, onnx_od, read_cfg, Findparent, deal_unlabeled_sample, imshow_det

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
        self.df = pd.DataFrame(columns=['path_img', 'labels'])

    def onnx_inference(self, path_imgs, onnx_map):
        """ONNX推理
        Args: 
            onnx_map: 父级或子级参数
        Returns: DataFrame存储所有图片的预标注结果
        """
        pool = Pool(self.nproc)
        for m in onnx_map:
            pool.apply_async(
                func=onnx_od, 
                args=(self.path_imgs, m, ), 
                callback=self.callback_merge_labels)
        pool.close()
        pool.join()

    def callback_merge_labels(self, result):
        """预标注结果合并
        Args:
            result: 预标注结果。key为path_img, value为图片对应的labels
        Returns: 
        """
        for path_img, labels in result.items():
            if not len(labels): print(f"{path_img}中不存在目标！")
            if path_img not in self.df['path_img'].values:
                self.df.loc[len(self.df)] = [path_img, labels]
            else:
                index = self.df['path_img'][self.df['path_img'].values == path_img].index
                self.df.loc[index, 'labels'].append(labels)

    def generate_single_json(self, info, labels):
        """预标注结果写入
        Args:
            info:   图片基本信息
            labels: 预标结果
        Returns: 
        """
        path_img = info["file"]
        path_json = info["json"].replace(self.inputInfo["markDataInPath"], self.inputInfo["markDataOutPath"])
        image = cv2.imread(path_img)
        H, W, _ = image.shape
        output_info = {
            "baseId": info["baseId"],
            "fileName": osp.basename(path_img),
            "imgSize": [W, H],
            "objects": []
        }
        obj_idx = 1
        labels_od2 = []
        for label in labels:
            x0, y0, x1, y1, cls = label
            obj_json = {
                    "class": cls,
                    "parent": [],
                    "coord": [[x0, y0], [x1, y1]],
                    "id": obj_idx,
                    "shape": "rect",
                    "props": {}
                }
            output_info["objects"].append(obj_json)
            obj_idx += 1
        return path_img, path_json, output_info, labels_od2
        
    def callback_merge_json(self, path_img, path_json, output_info, labels_od2):
        """一级od和二级od嵌套
        Args:
            path_img: 预标注图片路径
            path_json: 预标注json路径
            output_info: 一级od标注信息
            labels_od2: 二级od标注结果
        Returns: 
        """
        if not len(labels_od2):
            fp = Findparent()
            output_info = fp(path_img, output_info, labels_od2)

        fileDir = osp.dirname(path_json)
        if not osp.exists(fileDir): os.makedirs(fileDir)
        with open(path_json, 'w') as json_f:
            json.dump(output_info, json_f, indent=2, cls=NpEncoder)
        json_f.close()
        
    def run(self):
        cfg = Config.fromfile('./config.py')
        prelabeling_map = cfg.get('prelabeling_map', None)
        assert prelabeling_map, print("检查prelabeling_map配置！")
        parent_map, child_map = read_cfg(prelabeling_map)
        # 一级OD
        self.onnx_inference(self.path_imgs, parent_map)

        # 二级OD
        if len(child_map):
            self.onnx_inference(osp.join(osp.dirname(self.path_imgs[0]), "crop_imgs"), child_map)

        # json嵌套
        pool = Pool(self.nproc)
        for info in self.info:
            path_img = info["path_img"]
            index = self.df['path_img'][self.df['path_img'].values == path_img].index
            labels = self.df.loc[index, 'labels']
            pool.apply_async(
                func=self.generate_single_json, 
                args=(info, labels, ),
                callback=self.callback_merge_json
            )
        pool.close()
        pool.join()

        # 线下预标开启看效果，以及没有标注结果的情况
        if bool(self.inputInfo["args"]["show_result"]):
            deal_unlabeled_sample(self.path_imgs, self.inputInfo["markDataOutPath"])
            for path_img in self.path_imgs:
                bboxes = [label[:4] for label in self.df[path_img]]
                classes = set([label[4] for label in self.df[path_img]])
                labels = [classes.index(label[4]) for label in self.df[path_img]]
                imshow_det(path_img, bboxes, labels, save_path=osp.join(osp.dirname(osp.dirname(path_img)), 'show'))

if __name__ == '__main__':
    parser = getArgs()
    args = parser.parse_args()
    p = Prelabeling(args)
    p.run()
