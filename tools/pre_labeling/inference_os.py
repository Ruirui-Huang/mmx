import warnings, sys, json, time, os, argparse
warnings.filterwarnings("ignore")
import os.path as osp
import onnxruntime as ort
import onnx
import numpy as np
import multiprocessing 
import cv2
import pycocotools.mask as mask_util
from tqdm import tqdm


def getArgs():
    parser = argparse.ArgumentParser(description="语义分割预标注！")
    parser.add_argument('--token', type=str, required=True)
    parser.add_argument('--jsonFile', type=str, required=True)
    parser.add_argument('--filesInfo', type=str, required=True)
    return parser

parser = getArgs()
args = parser.parse_args()
jsonFile = open(args.jsonFile, mode='r', encoding='utf-8')
inputInfo = json.load(jsonFile)
fileInfo = open(args.filesInfo, "r", encoding="utf-8")
data = fileInfo.readlines()
p_bar = tqdm(data, ncols=100)
p_bar.set_description('Processing')

def multi_processing_pipeline(single_func, task_list, n_process=None, callback=None, **kw):
    pool = multiprocessing.Pool(processes=n_process)
    
    process_pool = []
    for i in range(len(task_list)):
        process_pool.append(
            pool.apply_async(single_func, args=(task_list[i], ), kwds=kw, callback=callback)
        )
    pool.close()
    pool.join()
    p_bar.close()
    print('success!')
    return process_pool

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


class PreLabeling():
    def __init__(self):
        cls_show = inputInfo["args"]['cls_show']
        self.classes = cls_show['classes']
        self.show = cls_show['show']
        self.num_classes = len(self.classes)
        assert self.num_classes == len(self.show), print("num_classes和show长度不一致！")
        self.maskId = np.random.randint(0, 255, self.num_classes)
        self.colors = [np.random.randint(0, 255, 3) for _ in range(self.num_classes)]

    def ImagePreProcess(self, path_img, img_size):
        """数据预处理
        Args:
            path_img: 图片路径
        Returns: 预处理后的图像数据
        """
        self.img = cv2.imread(path_img)
        self.H, self.W, _ = self.img.shape
        # 图像缩放
        resized_img = cv2.resize(self.img, img_size)
        # 归一化
        mean = np.array([0, 0, 0], dtype=np.float32)
        std = np.array([255, 255, 255], dtype=np.float32)
        resized_img = (resized_img - mean) / std
        return resized_img
        
    def OnnxInference(self, path_img):
        """ONNX前向
        Args:
            path_img: 图片路径
        Returns: 原图坐标系下的分割掩码结果
        """
        path_onnx = inputInfo["args"]['path_onnx']
        assert osp.exists(path_onnx), print(f"{path_onnx}不存在！")
        sess = ort.InferenceSession(path_onnx, providers=['CUDAExecutionProvider'])

        img_size = sess.get_inputs()[0].shape[-2:]
        input_name = sess.get_inputs()[0].name

        image = self.ImagePreProcess(path_img, img_size)
        image = image.transpose(2, 0, 1).astype(np.float32)
        
        sess_output = []
        for out in sess.get_outputs():
            sess_output.append(out.name)

        outputs = sess.run(output_names=sess_output, input_feed={input_name: [image]})[0][0] # C, W, H
        outputs = outputs.transpose(1, 2, 0) # W, H, C
        outputs = cv2.resize(outputs, [self.W, self.H]) # H, W, C
        return np.argmax(outputs, 2) # H, W

    def GenerateMask(self, result, path_img):
        """分割结果可视化
        Args:
            result: 原图坐标系下的分割掩码结果
            path_img: 图片路径
        Returns: 
        """
        index_list = np.unique(result)
        mask = np.zeros((self.H, self.W)).astype(np.uint8)
        color_seg = np.zeros([self.H, self.W, 3]).astype(np.uint8)
        for index in index_list:
            mask[result == index] = index
            color_seg[result == index] = self.colors[index]
        path_mask = osp.splitext(path_img.replace('/data/', '/mask/'))[0] +  ".png"
        filedir = osp.dirname(path_mask)
        if not osp.exists(filedir): os.makedirs(filedir)
        cv2.imwrite(path_mask, mask)

        opacity = 0.8
        img = self.img * (1-opacity) + color_seg[..., ::-1] * opacity
        img = np.concatenate((img, np.ones((self.H, self.W//50, 3), dtype=np.uint8)*255, self.img), axis=1)
        path_show = path_img.replace('/data/', '/show/')
        filedir = osp.dirname(path_show)
        if not osp.exists(filedir): os.makedirs(filedir)
        cv2.imwrite(path_show, img)
        
    def GenerateJson(self, result, info):
        """预标注结果写入
        Args:
            result: 原图坐标系下的分割掩码结果
            info: 预标注信息
        Returns: 
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
                        "coord": [[0, 0], [self.W, self.H]],
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
            json.dump(output_json_info, json_f, indent=2, cls=NpEncoder)
        json_f.close()
        
    def process(self, info):
        info = json.loads(info)
        path_img = info["file"]
        if osp.splitext(path_img)[-1] != '.jpg': return
        pred = self.OnnxInference(path_img)
        self.GenerateJson(pred, info)
        # self.GenerateMask(pred, path_img)
        
    def callback(self, event):
        p_bar.update()
        
    def run(self, n_process):
        multi_processing_pipeline(self.process, data, n_process=n_process, callback=self.callback)        
          
def main():
    start = time.time()
    p = PreLabeling()
    p.run(n_process=inputInfo["args"]["n_process"])
    jsonFile.close()
    fileInfo.close()
    end = time.time()
    print(f"预标注图片共{len(data)}张；耗时：{end-start}")

if __name__ == '__main__':
    main()
