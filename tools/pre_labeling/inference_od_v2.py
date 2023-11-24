# -*- coding: utf-8 -*-
import os, shutil, argparse, json, warnings
warnings.filterwarnings("ignore")
import cv2
from tqdm import tqdm
from multiprocessing import Pool
import os.path as osp
import pandas as pd
from mmengine.config import Config
from utils import NpEncoder, onnx_od

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
        self.info = fileInfo.readlines()
        self.inputInfo = json.load(jsonFile)
        jsonFile.close()
        fileInfo.close()
        self.nproc = inputInfo["args"]["n_process"]
        self.path_imgs = [info["file"] for info in self.info]
        self.df = pd.DataFrame(columns=['path_img', 'labels'])
    
    def callback(self, result):
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

    def onnx_inference(self):
        """ONNX推理
        Args:
        Returns: 
        """
        pool = Pool(self.nproc)
        for order in self.order:
            path_onnx = osp.join("./model", order)
            pool.apply_async(
                func=onnx_od, 
                args=(
                    self.path_imgs, 
                    path_onnx, 
                    model_type,
                    class_show,
                    anchors,
                    score_thr,
                    iou_thr,
                    ), 
                callback=self.callback)
        pool.close()
        pool.join()

    def generate_single_json(self, info):
        """预标注结果写入
        Args:
            info: 预标注信息
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
        index = self.df['path_img'][self.df['path_img'].values == path_img].index
        for label in self.df.loc[index, 'labels']:
            x0, y0, x1, y1, cls = label
            obj_json = {
                    "props": {},
                    "coord": [[x0, y0], [x1, y1]],
                    "class": cls,
                    "parent": [],
                    "id": self.obj_idx,
                    "shape": "poly"  
                }
            output_info["objects"].append(obj_json)
            obj_idx += 1
        
        fileDir = osp.dirname(path_json)
        if not osp.exists(fileDir): os.makedirs(fileDir)
        with open(path_json, 'w') as json_f:
            json.dump(output_info, json_f, indent=2, cls=NpEncoder)
        json_f.close()
        
    def deal_unlabeled_sample(self):
        """处理没有标注的数据
        Args:
        Returns: 
        """
        save_path = osp.join(osp.dirname(self.path_imgs), 'unlabeld')
        if not osp.exists(save_path): os.makedirs(save_path)
        files = os.listdir(self.path_imgs)
        p_bar = tqdm(files)
        for img in files:
            if osp.splitext(img)[-1] != '.jpg': continue
            path_img = osp.join(self.path_imgs, img)
            path_json = path_img + '.json'
            if not osp.exists(path_json):
                shutil.move(path_img, save_path)
            p_bar.update()
        p_bar.close()

    def run(self):
        # 一级od
        self.onnx_inference()
         # 二级od嵌套
        if order_od2:
            path_imgs = self.path_imgs
            self.path_imgs = path_crop_img
            self.order = 
            self.onnx_inference()
            fp = find_parent(path_imgs, path_crop_img, path_imgs)
            fp.run()

        pool = Pool(self.nproc)
        for info in self.info:
            pool.apply_async(
                func=self.generate_single_json, args=info)
        pool.close()
        pool.join()
        self.deal_unlabeled_sample()

if __name__ == '__main__':
    parser = getArgs()
    args = parser.parse_args()
    p = Prelabeling()
    p.run(args)