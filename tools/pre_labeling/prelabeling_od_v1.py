
# -*- coding: utf-8 -*-
import os, shutil, glob, argparse, sys, warnings
from tqdm import tqdm
from multiprocessing import Pool
import os.path as osp
import pandas as pd
from mmengine.config import Config
sys.path.append(osp.dirname(osp.dirname(osp.realpath(__file__))))
from mmconvertor.mmcvt import Runner
from mmconvertor.tools import New2old, Old2new, Videodirs2image, Merge_coco, Txt2CoCo, Nested
warnings.filterwarnings("ignore")

class Prelabeling:
    def __init__(self, cfg):
        self.cfg = cfg
        self.nproc = self.cfg.nproc
        try:
            self.path_imgs = self.cfg.path_imgs
        except:
            video_path = self.cfg.video.video_path
            if osp.isdir(video_path):
                self.path_imgs = osp.join(video_path, 'images')
            else:
                self.path_imgs = osp.splitext(video_path)[0]
            print("\n 抽帧中......")
            videodirs2image = Videodirs2image(
                video_path=video_path,
                out_type="one_dir",
                image_dir=self.path_imgs,
                fram_interval=self.cfg.video.fram_interval,
                sence_threshold = self.cfg.video.sence_threshold,
                filename_tmpl='%05d.jpg',
                filestart=0,
                start=0,
                max_num=0,
                show_progress=True,
                nproc=self.nproc,
                ext=('.dav','.avi',".mp4", ".wmv"),
                recursive=True,
                max_distance_threshold=self.cfg.video.max_distance_threshold)
            videodirs2image.run()
            
        self.label_map = dict()
        self.label_map_dino = dict()
        for key, value in self.cfg.label_map.items():
            label, caption = value.split('.')
            self.label_map_dino[caption] = key.split('.')[-1]
            self.label_map[key.split('.')[-1]] = label
        
    def predict_single_category(self, model_name):
        """单类别预标注     
        Args:
            model_name: 模型名称
        Returns: 
            path_single_txt: 单类别预测结果存储路径
        """
        print(f"\n 正在打印指令 {model_name} 的结果......")
        config = glob.glob(self.cfg.path_model + f"/{model_name}/*.py")[0]
        checkpoint = glob.glob(self.cfg.path_model + f"/{model_name}/best_coco/*.pth")[0]
        path_single_txt = os.path.join(self.root_imgs, f'{self.floder}_{model_name}.txt')
        commond = f'python utils/image_demo.py \
        {self.path_imgs} \
        {config} \
        {checkpoint} \
        --path-txt {path_single_txt} \
        --score-thr="{self.cfg.tta.score_thr}" \
        --scale-factors="{self.cfg.tta.scale_factors}" \
        --probs="{self.cfg.tta.probs}" \
        --directions="{self.cfg.tta.directions}"'
        os.system(commond)
        return path_single_txt
    
    def callback(self, path_single_txt):
        """预测结果合并写入DataFrame
        Args:
            path_single_txt: 单类别预测结果存储路径
        Returns: 
        """
        fr = open(path_single_txt, 'r')
        lines = fr.readlines()
        fr.close()
        for line in lines:
            tmp = line.strip('\n').split(' ')
            path_img, labels = tmp[0], tmp[1:]
            if len(labels) > 0:
                if path_img not in self.df['path_img'].values:
                    self.df.loc[len(self.df)] = [path_img, ' '.join(labels)]
                else:
                    index = self.df['path_img'][self.df['path_img'].values == path_img].index
                    self.df.loc[index, 'labels'] += ' ' + ' '.join(labels)
            else:
                print(f"\n {path_img}中不存在目标，请在{path_single_txt}中确认！")

    def txt_to_coco(self, path_imgs=None, order=None, classes=None, is_crop=None):
        """.txt生成CoCo文件
        Args:
            path_imgs:图片路径
            order:        需要预标注的目标
            class:          Dahuajson中的class名称
            is_crop:     是否涉及二级OD
        Returns: 
        """
        if path_imgs: self.path_imgs = path_imgs
        self.order = order
        self.classes = classes
        self.is_crop = is_crop
        self.root_imgs, self.floder = os.path.split(self.path_imgs)
        self.path_txt = os.path.join(self.root_imgs, f'{self.floder}.txt') 
        self.df = pd.DataFrame(columns=['path_img', 'labels'])
        pool = Pool(self.nproc)
        for order in self.order:
            pool.apply_async(
                func=self.predict_single_category, 
                args=(order, ), 
                callback=self.callback)
        pool.close()
        pool.join()
        fw = open(self.path_txt, 'w+')
        for _, row in self.df.iterrows():
            fw.write(row['path_img'] + ' ' + row['labels'] + '\n')
        fw.close()

        # 引入大模型
        if self.cfg.Bigmodel_cfg.state:
            commond = f'python utils/detector_sam_demo.py \
            {self.path_txt} \
            --det-config {self.cfg.Bigmodel_cfg.det_config} \
            --det-weight {self.cfg.Bigmodel_cfg.det_weight} \
            --weights {self.cfg.Bigmodel_cfg.weights} \
            --box-thr 0.3 \
            --text-thr 0.25 \
            --label-map="{self.label_map_dino}"'
            os.system(commond)

        print(f"\n .txt 转 coco中......")
        txt2coco = Txt2CoCo(self.path_txt, list(self.classes))
        txt2coco.forward()
        
    # 处理多级嵌套
    def get_jinn_classes(self):
        classes = []
        for label_str in self.label_map.values():
            section = label_str.split(".")
            cls = [section[0]]
            while len(section) > 1: 
                section.pop(0)
                sub_section = section[0].split("_")
                cls.append(sub_section[-1])
            classes.append(".".join(cls))
        return classes

    def transform_dahuajson(self):
        path_crop_img = osp.join(self.root_imgs, "crop_img")
        dataset = dict(
            type="Coco", 
            cfg=dict(
                img_prefix="", 
                ann_files=self.path_imgs + '.json',
                is_crop=self.is_crop,
                cropimg_save_path=path_crop_img)
        )    
        transform_cfg = dict(format="Dahua", nproc=self.nproc)
        save_path = osp.join(self.root_imgs, 'json_old')
        save_config = dict(save_path=save_path)
        hooks = [
            dict(
                type="Merge", 
                merge_file=True, 
                merge_bbox=True, 
                iou_thres=0.8,),
            dict(
                type="LabelMap", 
                label_map=self.label_map,),
            dict(
                type="DrawResults", 
                state=True if self.cfg.save_result else False, 
                classes=self.get_jinn_classes(),
                font_scale=1, 
                save_path=osp.join(self.root_imgs, self.cfg.save_result) if self.cfg.save_result else None),
        ]
        print(f"\n coco 转 dahua_old中......")

        runner = Runner(dataset, transform_cfg)
        check_config = self.cfg.get('check_config', None)
        runner.register_transform_hook(save_config=save_config, check_config=check_config)
        runner.register_custom_hooks(hooks=hooks)
        runner.run()

        Old2new.transform(save_path, '', self.path_imgs)
        shutil.rmtree(save_path)
        return self.path_imgs, path_crop_img

    # 处理无效目标的样本
    def deal_inv_samples(self):
        path_unlabeld = osp.join(osp.dirname(self.path_imgs), self.cfg.path_unlabeld)
        if not osp.exists(path_unlabeld): 
            os.makedirs(path_unlabeld)
        files = os.listdir(self.path_imgs)
        p_bar = tqdm(files)
        for img in files:
            if osp.splitext(img)[-1] != '.jpg': continue
            path_img = osp.join(self.path_imgs, img)
            path_json = path_img + '.json'
            if not osp.exists(path_json):
                shutil.move(path_img, path_unlabeld)
            p_bar.update()
        p_bar.close()

def parse_args():
    parser = argparse.ArgumentParser(description='Pre_labeling......')
    parser.add_argument('config', help='adapter config file path')
    args = parser.parse_args()
    return args 

def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)
    pre_labeling = Prelabeling(cfg)

    order = pd.DataFrame(columns=['OD1', 'OD2', 'cls'])
    for i, key in enumerate(cfg.label_map.keys()): 
        order.loc[i] = key.split('.')

    order_od1 = set(order[order.OD2 == 'None'].OD1)
    classes_od1 = set(order[order.OD2 == 'None'].cls)
    is_crop = set(order[~(order.OD2 == 'None')].OD1)
    order_od2 = set(order[~(order.OD2 == 'None')].OD2)
    classes_od2 = set(order[~(order.OD2 == 'None')].cls)

    # step1：单类别预测 -> 标注合并 -> 转coco
    pre_labeling.txt_to_coco(order=order_od1, classes=classes_od1, is_crop=is_crop)

    # step2：转dahua
    path_imgs, path_crop_img = pre_labeling.transform()

    # step3：二级od嵌套
    if order_od2:
        pre_labeling.txt_to_coco(path_imgs=path_crop_img, order=order_od2, classes=classes_od2)
        pre_labeling.transform()

        print(f"\n 合并后的json位于{path_imgs}\n")
        nested = Nested(path_imgs, path_crop_img, path_imgs)
        nested.nesting_level()
        
    # step4：数据过滤
    if cfg.path_unlabeld:
        pre_labeling.deal_inv_samples()

if __name__ == '__main__':
    main()