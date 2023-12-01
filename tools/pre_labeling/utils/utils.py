# -*- coding: utf-8 -*-
import json, os, shutil, copy, glob
import os.path as osp
import numpy as np
from tqdm import tqdm
import multiprocessing

def multi_processing_pipeline(single_func, task_list, n_process=None, callback=None, **kw):
    """基于流水线思想的多进程处理。将单个任务放入进程池中，由系统调度哪个进程处理该任务
    Args:
        single_func: 单进程处理函数, 传入单个任务
        task_list: 任务列表
        n_process: 进程的数量，当n_process为None时，表示使用全部cpu核心
    """
    if n_process:
        n_process = n_process
    else:
        n_process = os.cpu_count()

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
    
class Npencoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

def read_cfg(prelabeling_map, args):
    """模型库解析拆分一级OD和二级OD
    Args:
        prelabeling_map (dict): 模型库。以onnx名称作为键值，存储onnx的详细信息
        args (dict): 需要预标的目标类别信息
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
    classes_parent = set()
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
                    "is_show": [0]*num_classes,
                    "exist_child": [0]*num_classes,
                }
                child[onnx_name]["Class_show"] = {
                    "classes": [f"obj{i}" for i in range(num_classes)], 
                    "is_show": [0]*num_classes,
                    "exist_child": [0]*num_classes,
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
                    classes_parent.add(value["Parent"][index])


    parent_map, child_map = [], []
    for _, map in parent.items():
        if not sum(map["Class_show"]["is_show"]): continue
        map["Class_show"]["exist_child"] = [int(i in classes_parent) for i in map["Class_show"]["classes"]]
        parent_map.append(map)

    for _, map in child.items():
        if not sum(map["Class_show"]["is_show"]): continue
        child_map.append(map)

    return parent_map, child_map

def deal_unlabeled_sample(path_imgs, path_jsons=None, remove=False, save_path=None):
    """处理没有标注的数据
    Args: 
        path_imgs (str): 图片路径
        path_json (str): 预标注json路径
        remove (bool): 移动到save_path或者直接删除
        save_path (str): 存储没有标注结果的图片
    Returns: 
    """
    if isinstance(path_imgs, str): path_imgs = glob.glob(osp.join(path_imgs, "*.jpg"))
    p_bar = tqdm(path_imgs, ncols=100)
    p_bar.set_description("Unlabeled data Processing")
    for path_img in path_imgs:
        p_bar.update()
        if path_jsons: 
            path_json = osp.join(path_json, osp.split(path_img)[-1]).replace('.jpg', '.json')
        else:
            path_json = path_img.replace('.jpg', '.json')
        if osp.exists(path_json): continue
        if remove:
            os.remove(path_img)
        else:
            if not save_path:
                save_path = osp.join(osp.dirname(path_img), 'unlabeld')
            if not osp.exists(save_path): 
                os.makedirs(save_path)
            shutil.move(path_img, save_path)
    p_bar.close()
