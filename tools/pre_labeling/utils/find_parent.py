import os
import json
import os.path as osp
from PIL import Image
import numpy as np

class Findparent():
    def __init__(self, path_parent_json, path_child_json, save_path):
        self.path_parent_json = path_parent_json
        self.path_child_json = path_child_json
        self.save_path = save_path
        self.info = dict()
    
    # 确定crop_img属于哪一张大图
    def get_parent_children(self):
        for json in os.listdir(self.path_parent_json):
            if '.jpg.json' in json:
                name = json.split('.jpg.json')[0]
                parent_json = osp.join(self.path_parent_json, json)
                self.info[parent_json] = []
                num = 1
                while True:
                    child_json = osp.join(self.path_child_json, name + f'_crop{num}.jpg.json')
                    if osp.exists(child_json):
                        self.info[parent_json].append(child_json)
                        num += 1
                    else: break   

    # 获取每张大图上的检测信息，为后续明确父级做准备
    def get_parent_info(self, parent_file):
        parent_image = Image.open(osp.join(self.path_parent_json, parent_file['fileName']))
        parent_info = dict()
        parent_info['id'] = []
        parent_info['x0y0'] = []
        parent_info['img'] = []
        for i in parent_file['objects']:
            parent_info['id'].append(i['id'])
            [[xmin, ymin], [xmax, ymax]] = i['coord']
            parent_info['x0y0'].append([xmin, ymin])
            parent_info['img'].append(parent_image.crop([xmin, ymin, xmax, ymax]))
        return parent_info

    # 相似度度量，确定crop_img在大图上的具体位置
    def compare(self, parent_img, child_img):
        parent_img = np.array(parent_img.resize((100, 100)).convert('L'))
        child_img = np.array(child_img.resize((100, 100)).convert('L'))
        MSE = sum(sum(parent_img-child_img)**2)/100**2
        return 1 - MSE

    # 获取crop_img的父级id，和父级左上角坐标
    def get_parent_id(self, parent_info, child_file):
        best_ind, best_sim = 0, 0
        child_image = Image.open(osp.join(self.path_child_json, child_file['fileName']))
        for ind, img in enumerate(parent_info['img']):
            sim = self.compare(img, child_image)
            if sim > best_sim:
                best_ind = ind
                best_sim = sim
        return parent_info['id'][best_ind], parent_info['x0y0'][best_ind]
                     
    # 嵌套二级OD到一级OD
    def run(self):
        self.get_parent_children()
        for parent_json, children in self.info.items():
            ann_id = len(children)
            if ann_id == 0: return
            fr_parent = open(parent_json, 'r')
            parent_file = json.load(fr_parent)
            parent_info = self.get_parent_info(parent_file)
            for child_json in children:
                fr_child = open(child_json, 'r')
                child_file = json.load(fr_child)
                parent_id, xoyo = self.get_parent_id(parent_info, child_file)
                for i in child_file['objects']: 
                    xmin, ymin = xoyo
                    i['parent'] = [parent_id]
                    ann_id += 1
                    i['id'] = ann_id
                    tmp = i['coord']
                    i['coord'] = [[tmp[0][0]+xmin, tmp[0][1]+ymin], [tmp[1][0]+xmin, tmp[1][1]+ymin]] 
                    parent_file['objects'].append(i)
            save_path = osp.join(self.save_path, osp.split(parent_json)[-1])
            fw = open(save_path, 'w')
            json.dump(parent_file, fw, ensure_ascii=False, indent=2)
            fw.close()
                

if __name__ == '__main__':
    path_parent_json = "/extraStore/secondStorePath/groupdata/ocralgorithm/Mars/Venom/Venom1/02_data/02_train/HuangR/TEST/0220"
    path_child_json = "/extraStore/secondStorePath/groupdata/ocralgorithm/Mars/Venom/Venom1/02_data/02_train/HuangR/TEST/crop_img"
    fp = Findparent(path_parent_json, path_child_json, path_parent_json)
    fp.run()