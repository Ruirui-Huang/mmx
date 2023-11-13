# Copyright (c) OpenMMLab. All rights reserved.
import os, sys
import os.path as osp
from argparse import ArgumentParser

import mmcv
from mmdet.apis import inference_detector, init_detector
from mmengine.logging import print_log
from mmengine.utils import ProgressBar

from mmyolo.registry import VISUALIZERS
from mmyolo.utils import register_all_modules, switch_to_deploy
from mmyolo.utils.misc import get_file_list
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))

def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        'img', help='Image path, include image file, dir and URL.')
    parser.add_argument('config', help='Config file')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument(
        '--out-dir', default='./output', help='Path to output file')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--show', action='store_true', help='Show the detection results')
    parser.add_argument(
        '--is-crop', default=False, action='store_true', help='Crop the boxes')
    parser.add_argument(
        '--deploy',
        action='store_true',
        help='Switch model to deployment mode')
    parser.add_argument(
        '--score-thr', type=float, default=0.3, help='Bbox score threshold')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    # register all modules in mmdet into the registries
    register_all_modules()

    # build the model from a config file and a checkpoint file
    model = init_detector(args.config, args.checkpoint, device=args.device, cfg_options={})

    if args.deploy:
        switch_to_deploy(model)

    if not os.path.exists(args.out_dir) and not args.show:
        os.mkdir(args.out_dir)

    # init visualizer
    visualizer = VISUALIZERS.build(model.cfg.visualizer)
    visualizer.dataset_meta = model.dataset_meta

    # get file list
    files, source_type = get_file_list(args.img)

    # start detector inference
    progress_bar = ProgressBar(len(files))
    tmp = []
    for file in files:
        if source_type['is_dir']:
            filename = os.path.relpath(file, args.img).replace('/', '_')
        else:
            filename = os.path.basename(file)
        out_file = None if args.show else os.path.join(args.out_dir, filename)
        # if osp.exists(out_file): continue
        result = inference_detector(model, file)

        image = mmcv.imread(file)
        img = mmcv.imconvert(image, 'bgr', 'rgb')

        boxes = result.pred_instances.bboxes
        for box in boxes:
            xmin, ymin, xmax, ymax = [int(i) for i in box]
            if args.is_crop:
                try:
                    image_crop = image[ymin: ymax, xmin: xmax]
                    mmcv.imwrite(image_crop, osp.join(args.out_dir+'/crop', osp.splitext(filename)[0] + f'_{xmin}_{ymin}_{xmax}_{ymax}.jpg'))
                except: pass
            tmp.append(min(ymax-ymin, xmax-xmin))

        # 仅展示部分目标
        # result.pred_instances = result.pred_instances[result.pred_instances.labels == 3]

        progress_bar.update()
        visualizer.add_datasample(
            os.path.basename(out_file),
            img,
            data_sample=result,
            draw_gt=False,
            show=args.show,
            wait_time=0,
            out_file=out_file,
            pred_score_thr=args.score_thr)

    if not args.show:
        print_log(
            f'\nResults have been saved at {os.path.abspath(args.out_dir)}')

    # import numpy as np
    # import matplotlib.pyplot as plt
    # plt.boxplot(np.array(tmp), sym='o', whis=1.5)
    # plt.savefig(osp.join(args.out_dir, "plot.jpg"))


if __name__ == '__main__':
    main()
