import warnings, json, time, os, argparse, sys
warnings.filterwarnings("ignore")
import cv2
import os.path as osp
import numpy as np
from tqdm import tqdm
import onnxruntime as ort
import pycocotools.mask as mask_util
from utils import multi_processing_pipeline, imshow_semantic, Preprocess, Npencoder

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
        cls_show = inputInfo["args"]['cls_show']
        self.classes = cls_show['classes']
        self.show = cls_show['show']
        self.num_classes = len(self.classes)
        assert self.num_classes == len(self.show), print("num_classes和show长度不一致！")
        self.maskId = np.random.randint(0, 255, self.num_classes)
        
    def onnx_inference(self, path_img, path_onnx):
        """ONNX前向
        Args:
            path_img (str): 图片路径
            path_onnx (str): onnx路径
        """
        preprocessor = Preprocess()
        assert osp.exists(path_onnx), print(f"{path_onnx}不存在！")
        # 加载引擎
        sess = ort.InferenceSession(path_onnx, providers=['CUDAExecutionProvider'])
        # 数据预处理
        image = cv2.imread(path_img)
        input_size = sess.get_inputs()[0].shape[-2:]
        self.H, self.W, _ = image.shape
        input_data, (_, _), _ = preprocessor(image, input_size)
        input_name = sess.get_inputs()[0].name
        # 推理
        outputs = sess.run(output_names=None, input_feed={input_name: input_data})[0][0] # C, W, H
        # 后处理
        outputs = outputs.transpose(1, 2, 0) # W, H, C
        outputs = cv2.resize(outputs, [self.W, self.H]) # H, W, C
        return np.argmax(outputs, 2) # H, W
        
    def generate_json(self, result, info):
        """预标注结果写入Dahuajson
        Args:
            result (ndarray): 原图坐标系下的分割结果
            info (dict):   预标注基本信息
        """
        output_json_info = {
            "baseId": info["baseId"],
            "fileName": osp.basename(info["file"]),
            "imgSize": [self.W, self.H],
            "objects": []
        }
        obj_idx = 1
        for idx in range(1, self.num_classes):
            if not self.show[idx]: continue
            mask = np.zeros((self.H, self.W)).astype(np.uint8)
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
                        "class": self.classes[idx],
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
                        "coord": [[0, 0], [self.W, self.H]],  # 此处可以额外计算各类别外接矩形框作为coord传入
                        "class": self.classes[idx],
                        "area": area,
                        "parent": [],
                        "id": int(obj_idx),
                        "rle": rle,
                        "maskId": self.maskId[idx],
                        "shape": "mask"
                    }
                output_json_info["objects"].append(obj_json)
                obj_idx += 1

        jsonSavePath = info["json"].replace(inputInfo["markDataInPath"], inputInfo["markDataOutPath"])
        fileDir = osp.dirname(jsonSavePath)
        if not osp.exists(fileDir): os.makedirs(fileDir)
        with open(jsonSavePath, 'w') as json_f:
            json.dump(output_json_info, json_f, indent=2, cls=Npencoder)
        json_f.close()
        
    def process(self, info):
        info = json.loads(info)
        path_img = info["file"]
        path_onnx = osp.join("./model_zoo", inputInfo["args"]['onnx_name'] + ".onnx")
        if osp.splitext(path_img)[-1] != '.jpg': return
        pred = self.onnx_inference(path_img, path_onnx)
        self.generate_json(pred, info)

        # 线下预标需要开启看效果
        if bool(inputInfo["args"]["save_result"]):
            imshow_semantic(path_img, pred)
        
    def callback(self, event):
        p_bar.update()
        
    def run(self, n_process):
        multi_processing_pipeline(
            self.process, 
            data, 
            n_process=n_process, 
            callback=self.callback
        )
          
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