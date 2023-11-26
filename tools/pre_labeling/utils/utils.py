import json, os, shutil, copy
from enum import Enum
import os.path as osp
import numpy as np
from tqdm import tqdm
import multiprocessing 
from pathlib import Path

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)
    
def multi_processing_pipeline(single_func, task_list, n_process=None, callback=None, **kw):
    pool = multiprocessing.Pool(processes=n_process)
    process_pool = []
    for i in range(len(task_list)):
        process_pool.append(
            pool.apply_async(single_func, args=(task_list[i], ), kwds=kw, callback=callback)
        )
    pool.close()
    pool.join()
    print('success!')
    return process_pool

IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp')

def path_to_list(path: str):
    path = Path(path)
    if path.is_file() and path.suffix in IMG_EXTENSIONS:
        res_list = [str(path.absolute())]
    elif path.is_dir():
        res_list = [
            str(p.absolute()) for p in path.iterdir()
            if p.suffix in IMG_EXTENSIONS
        ]
    else:
        raise RuntimeError
    return res_list

def read_cfg(prelabeling_map, args):
    """模型库解析拆分一级OD和二级OD
    Args:
        prelabeling_map: 模型库。以onnx名称作为键值，存储onnx的详细信息
        args: 需要预标的目标类别
    Returns: 细化一级od和二级od
    """
    ModelType = [
        'yolov5', 
        'yolov6', 
        'yolov7',
        'yolov8'
        'yolox', 
        'rtmdet',
        'ppyoloe', 
        'ppyoloep'
    ]
    parent, child = {}, {}
    for cls, is_used in args.items():
        for onnx_name, value in prelabeling_map.items():
            if cls not in value["Used_classes"] or not is_used: continue
            num_classes = value["Num_classes"]
            assert value["Model_type"] in ModelType  and \
            value["Num_classes"] > 0 and \
            value["Score_thr"] > 0 and \
            value["Box_thr"] > 0 and \
            (value["Anchors"] == None or len(value["Anchors"]) == 3) and \
            len(value["Used_classes"]) <= num_classes and \
            len(value["Class_index"]) <= num_classes and \
            len(value["Class_index"]) == len(value["Used_classes"]) and \
            max(value["Class_index"]) < num_classes and \
            (value["Parent"] == None or len(value["Parent"]) == len(value["Used_classes"])), print("请检查config配置！")

            value["Path_onnx"] = osp.join("./model_zoo", onnx_name + ".onnx")
            if onnx_name not in parent.keys():
                child[onnx_name] = copy.copy(value)
                parent[onnx_name] = copy.copy(value)

            if "Class_show" not in child[onnx_name].keys():
                parent[onnx_name]["Class_show"] = {
                    "classes": [f"obj{i}" for i in range(num_classes)], 
                    "is_show": [0]*num_classes
                }
                child[onnx_name]["Class_show"] = {
                    "classes": [f"obj{i}" for i in range(num_classes)], 
                    "is_show": [0]*num_classes
                }

            index = value["Class_index"][value["Used_classes"].index(cls)]
            if value["Parent"] == None: 
                parent[onnx_name]["Class_show"]["classes"][index] = cls
                parent[onnx_name]["Class_show"]["is_show"][index] = 1
            else:
                if value["Parent"][index] == None:
                    parent[onnx_name]["Class_show"]["classes"][index] = cls
                    parent[onnx_name]["Class_show"]["is_show"][index] = 1
                else:
                    child[onnx_name]["Class_show"]["classes"][index] = cls
                    child[onnx_name]["Class_show"]["is_show"][index] = 1

    parent_map, child_map = [], []
    for _, map in parent.items():
        if not sum(map["Class_show"]["is_show"]): continue
        parent_map.append(map)

    for _, map in child.items():
        if not sum(map["Class_show"]["is_show"]): continue
        child_map.append(map)

    return parent_map, child_map

def deal_unlabeled_sample(path_imgs, path_json, remove=False, path_save=None):
    """处理没有标注的数据
    Args: 
        path_imgs: 图片路径
        path_json: 预标注json路径
        remove: 移动到path_save或者直接删除
        path_save: 存储没有标注结果的图片
    Returns: 
    """
    files = path_to_list(path_imgs)
    p_bar = tqdm(files)
    for img in files:
        path_img = osp.join(path_imgs, img)
        path_json = osp.join(path_json, img) + '.json'
        if not osp.exists(path_json):
            if remove:
                os.remove(path_img)
            else:
                if not save_path:
                    save_path = osp.join(osp.dirname(path_img), 'unlabeld')
                if not osp.exists(save_path): 
                    os.makedirs(save_path)
                shutil.move(path_img, save_path)
        p_bar.update()
    p_bar.close()