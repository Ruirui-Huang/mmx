import warnings, json, time, os, argparse, sys
warnings.filterwarnings("ignore")
import cv2
import os.path as osp
import numpy as np
from tqdm import tqdm
import onnxruntime as ort
from multiprocessing import Pool
from mmengine.config import Config
import pycocotools.mask as mask_util
from utils import inference_os, read_cfg, imshow_semantic, Preprocess, Npencoder

def getArgs():
    """
    巨灵平台获取该方法参数的入口
    """
    parser = argparse.ArgumentParser(description="语义分割预标注！")
    parser.add_argument('--token', type=str, required=True)
    parser.add_argument('--jsonFile', type=str, required=True)
    parser.add_argument('--filesInfo', type=str, required=True)
    return parser

class PreLabeling():
    def __init__(self):
        cfg = Config.fromfile('./utils/config.py')
        prelabeling_map = cfg.get('prelabeling_map', None)
        assert prelabeling_map, print("检查prelabeling_map配置！")
        self.onnx_map, _ = read_cfg(prelabeling_map, inputInfo["args"])
        self.classes = []
        for m in self.onnx_map:
            classes = m["Class_show"]["classes"]
            is_show = m["Class_show"]["is_show"]
            for c, i in zip(classes, is_show):
                if i: self.classes.append(c)
        self.num_classes = len(self.classes)
        self.maskId = np.random.randint(0, 255, self.num_classes)
        self.save_result = False
        try:
            if bool(inputInfo["args"]["save_result"]):
                # 线下预标需要开启看效果
                self.save_result = bool(inputInfo["args"]["save_result"])
        except: pass
        
    def onnx_inference(self, info, path_onnx, class_show):
        """ONNX前向
        Args:
            info (str): 预标注基本信息
            path_onnx (str): onnx路径
            class_show (dict)
        """
        classes = class_show["classes"]
        is_show = class_show["is_show"]
        path_img = info["file"]
        image = cv2.imread(path_img)
        H, W, _ = image.shape
        path_json = info["json"].replace(inputInfo["markDataInPath"], inputInfo["markDataOutPath"])
        fileDir = osp.dirname(path_json)
        if not osp.exists(fileDir): os.makedirs(fileDir)
        obj_idx = 1
        if osp.exists(path_json):
            output_json_info = open(path_json, 'r')
            for obj in output_json_info["objects"]:
                    obj_idx = max(obj_idx, obj["id"])
        else:
            output_json_info = {
                "baseId": info["baseId"],
                "fileName": osp.basename(info["file"]),
                "imgSize": [W, H],
                "objects": []
            }
        result = inference_os(image, path_onnx)
        for idx, cls in enumerate(classes):
            if not is_show[idx]: continue
            mask = np.zeros((H, W)).astype(np.uint8)
            mask[result == idx] = 1
            mask = mask[:, :, None]
            # POLY标注
            if not bool(inputInfo["args"]["use_sam"]):
                contours = \
                    cv2.findContours(mask*255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
                for contour in contours:
                    polygon = contour.reshape(-1, 2)
                    if polygon.shape[0] < 3 or cv2.contourArea(contour) < 9:
                        continue
                    for i in range(len(polygon)):
                        polygon[i][0] = polygon[i][0]
                        polygon[i][1] = polygon[i][1]
                    obj_json = {
                        "props": {},
                        "coord": polygon,
                        "class": cls,
                        "parent": [],
                        "id": int(obj_idx),
                        "shape": "poly"  
                    }
                    output_json_info["objects"].append(obj_json)
                    obj_idx += 1
            # RLE标注
            else:
                area = mask.sum()
                if area < 10: continue
                rle = mask_util.encode(np.array(mask, order="F"))[0]
                rle = rle["counts"].decode("utf-8")
                obj_json = {
                        "props": {},
                        "coord": [[0, 0], [W, H]],  # 此处可以额外计算各类别外接矩形框作为coord传入
                        "class": cls,
                        "area": area,
                        "parent": [],
                        "id": int(obj_idx),
                        "rle": rle,
                        "maskId": self.maskId[self.classes.index(cls)],
                        "shape": "mask"
                    }
                output_json_info["objects"].append(obj_json)
                obj_idx += 1

        with open(path_json, 'w') as fw:
            json.dump(output_json_info, fw, indent=2, cls=Npencoder)
        fw.close()
        return [path_img, result]
    
    def callback(self, result):
        """类别合并
        Args:
            result (list): 组成为[图片路径, 预标结果] 
        """
        if not self.save_result: return
        

    def run(self, n_process):
        for m in self.onnx_map:
            path_model = m["Path_model"]
            print(f'{osp.split(path_model)[-1]} Processing: ')
            pool = Pool(n_process)
            for info in data:
                info = json.loads(info)
                pool.apply_async(
                    func=self.onnx_inference, 
                    args=(info, path_model, )
                    callback=self.callback)
          
def main():
    start = time.time()
    p = PreLabeling()
    p.run(n_process=inputInfo["args"]["n_process"])
    jsonFile.close()
    fileInfo.close()
    p_bar.close()
    end = time.time()
    print(f"预标注耗时：{end-start}s")


parser = getArgs()
args = parser.parse_args()
jsonFile = open(args.jsonFile, mode='r', encoding='utf-8')
inputInfo = json.load(jsonFile)
fileInfo = open(args.filesInfo, "r", encoding="utf-8")
data = fileInfo.readlines()
p_bar = tqdm(data, ncols=100)
p_bar.set_description('Processing')

if __name__ == '__main__':
    main()