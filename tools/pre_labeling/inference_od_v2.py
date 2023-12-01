# -*- coding: utf-8 -*-
import os, argparse, json, warnings
warnings.filterwarnings("ignore")
import cv2
import os.path as osp
import pandas as pd
from tqdm import tqdm
from multiprocessing import Pool
from mmengine.config import Config
from utils import onnx_od, read_cfg, deal_unlabeled_sample, imshow_det, Npencoder

def getArgs():
    """
    巨灵平台获取该方法参数的入口
    """
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
            onnx_map (dict): 父级或子级onnx参数
        """
        pool = Pool(self.nproc)
        for m in onnx_map:
            pool.apply_async(
                func=onnx_od, 
                args=(path_imgs, m, ), 
                callback=self.callback_merge_labels)
        pool.close()
        pool.join()

    def callback_merge_labels(self, result):
        """预标注结果合并
        Args:
            result (dict): 预标注结果。key为图片路径, value为图片对应的所有标注结果
        """
        for path_img, labels in result.items():
            if not len(labels): print(f"{path_img}中不存在目标！")
            if path_img not in self.df['path_img'].values:
                self.df.loc[len(self.df)] = [path_img, labels]
            else:
                index = self.df['path_img'][self.df['path_img'].values == path_img].index
                self.df.loc[index, 'labels'].values[0].extend(labels)
                
    def generate_single_json(self, info, labels):
        """预标注结果写入
        Args:
            info (dict):   图片基本信息
            labels (list): 预标结果，其中各元素的组成为[左上角横坐标, 左上角纵坐标, 右下角横坐标, 右下角纵坐标, 类别, 父级信息]
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
        parent_id_map = dict()
        for label in labels:
            x0, y0, x1, y1, cls, note = label
            if note: continue
            parent_id_map[f"{cls}_{x0}_{y0}_{x1}_{y1}"] = [cls, obj_idx]
            obj_json = {
                "class": str(cls),
                "parent": [],
                "coord": [[x0, y0], [x1, y1]],
                "id": obj_idx,
                "shape": "rect",
                "props": {}
            }
            output_info["objects"].append(obj_json)
            obj_idx += 1
        return [path_json, obj_idx, output_info, parent_id_map, labels]
        
    def callback_merge_json(self, result):
        """一级od和二级od嵌套
        Args:
            result (list): 组成为[标注路径, 一级od目标数量, 一级od标注信息, 二级od标注信息, 预标结果] 
        """
        obj_idx = result[1]
        for label in result[4]:
            x0, y0, x1, y1, cls, note = label
            if not note: continue
            cls_parent, value = osp.split("_")
            cls_parent, id_parent = result[3][note]
            obj_json = {
                "class": str(cls_parent),
                "parent": [id_parent],
                "coord": [[x0, y0], [x1, y1]],
                "id": obj_idx,
                "shape": "rect",
                "props": {cls_parent: value}
            }
            result[2]["objects"].append(obj_json)
            obj_idx += 1
        fileDir = osp.dirname(result[0])
        if not osp.exists(fileDir): os.makedirs(fileDir)
        with open(result[0], 'w') as json_f:
            json.dump(result[2], json_f, indent=2, cls=Npencoder)
        json_f.close()

    def show(self):
        """预标注结果可视化
        """
        p_bar = tqdm(self.df, ncols=100)
        p_bar.set_description('Drawing')
        for _, row in self.df.iterrows():
            p_bar.update()
            path_img = row['path_img']
            labels = row['labels']
            if not len(labels): continue
            bboxes = [label[:4] for label in labels]
            classes = list(set([label[4] for label in labels]))
            llabels = [classes.index(label[4]) for label in labels]
            imshow_det(
                path_img, bboxes, llabels, 
                classes=classes, 
                save_path=osp.join(osp.dirname(osp.dirname(path_img)), 'show')
            )
        p_bar.close()

    def run(self):
        cfg = Config.fromfile('./utils/config.py')
        prelabeling_map = cfg.get('prelabeling_map', None)
        assert prelabeling_map, print("检查prelabeling_map配置！")
        parent_map, child_map = read_cfg(prelabeling_map, self.inputInfo["args"])
        # 一级OD
        self.onnx_inference(self.path_imgs, parent_map)

        # 二级OD
        if len(child_map):
            self.onnx_inference(osp.join(osp.dirname(self.path_imgs[0]), "crop_imgs"), child_map)
        
        # 生成dahuajson
        pool = Pool(self.nproc)
        for info in self.info:
            path_img = info["file"]
            index = self.df['path_img'][self.df['path_img'].values == path_img].index
            labels = self.df['labels'].values[index]
            if not len(labels): continue
            pool.apply_async(
                func=self.generate_single_json, 
                args=(info, labels[0], ),
                callback=self.callback_merge_json
            )
        pool.close()
        pool.join()

        # 线下预标开启看效果，以及没有标注结果的情况
        if bool(self.inputInfo["args"]["show_result"]):
            # deal_unlabeled_sample(self.path_imgs)
            self.show()
            

if __name__ == '__main__':
    parser = getArgs()
    args = parser.parse_args()
    p = Prelabeling(args)
    p.run()