# Copyright (c) OpenMMLab. All rights reserved.
from collections import defaultdict
from pathlib import Path
from typing import Optional, Sequence, Union

import mmcv
import cv2
import torch
import numpy as np

from mmengine import Config
from mmengine.dataset import Compose
from mmengine.runner import load_checkpoint
from mmengine.registry import init_default_scope

from mmx.registry import MODELS

def init_model(config: Union[str, Path, Config],
               checkpoint: Optional[str] = None,
               device: str = 'cuda:0',
               cfg_options: Optional[dict] = None):
    """Initialize a model from config file.

    Args:
        config (str, :obj:`Path`, or :obj:`mmengine.Config`): Config file path,
            :obj:`Path`, or the config object.
        checkpoint (str, optional): Checkpoint path. If left as None, the model
            will not load any weights.
        device (str, optional) CPU/CUDA device option. Default 'cuda:0'.
            Use 'cpu' for loading model on CPU.
        cfg_options (dict, optional): Options to override some settings in
            the used config.
    Returns:
        nn.Module: The constructed model.
    """
    if isinstance(config, (str, Path)):
        config = Config.fromfile(config)
    elif not isinstance(config, Config):
        raise TypeError('config must be a filename or Config object, '
                        'but got {}'.format(type(config)))
    if cfg_options is not None:
        config.merge_from_dict(cfg_options)
    elif 'init_cfg' in config.model.backbone:
        config.model.backbone.init_cfg = None
    config.model.train_cfg = None

    init_default_scope(config.default_scope)

    config.model.setdefault('data_preprocessor', config.get('data_preprocessor', None))
    model = MODELS.build(config.model)
    if checkpoint is not None:
        checkpoint = load_checkpoint(model, checkpoint, map_location='cpu')
        dataset_meta = checkpoint['meta'].get('dataset_meta', None)
        # save the dataset_meta in the model for convenience
        if 'dataset_meta' in checkpoint.get('meta', {}):
            # mmseg 1.x
            model.dataset_meta = dataset_meta
        elif 'classes' in checkpoint.get('meta', {}):
            # < mmseg 1.x
            classes = checkpoint['meta']['classes']
            palette = checkpoint['meta']['palette']
            model.dataset_meta = {'classes': classes, 'palette': palette}
    try:
        config.custom_hooks.append(dict(type='mmyolo.SwitchToDeployHook'))
        model.cfg = config  # save the config in the model for convenience
    except: pass
    model.to(device)
    model.eval()
    return model


ImageType = Union[str, np.ndarray, Sequence[str], Sequence[np.ndarray]]


def _preprare_data(imgs: ImageType, model):

    cfg = model.cfg
    for pipeline in cfg.test_pipeline:
        if "Annotations" in pipeline["type"]:
            cfg.test_pipeline.remove(pipeline)
    # if dict(type='LoadAnnotations') in cfg.test_pipeline:
    #     cfg.test_pipeline.remove(dict(type='LoadAnnotations'))

    if isinstance(imgs, (list, tuple)):
        is_batch = True
    else:
        imgs = [imgs]
        is_batch = False

    if isinstance(imgs[0], np.ndarray):
        cfg.test_pipeline[0]['type'] = 'LoadImageFromNDArray'

    # TODO: Consider using the singleton pattern to avoid building
    # a pipeline for each inference
    pipeline = Compose(cfg.test_pipeline)

    data = defaultdict(list)
    #TODO: remove img_id.
    for img in imgs:
        if isinstance(img, np.ndarray):
            data_ = dict(img=img)
        else:
            data_ = dict(img_path=img)
        data_ = pipeline(data_)
        data['inputs'].append(data_['inputs'])
        data['data_samples'].append(data_['data_samples'])

    return data, is_batch


def inference_model(model, img):
    """Inference image(s) with the model.

    Args:
        model (nn.Module): The loaded model.
        imgs (str/ndarray or list[str/ndarray]): Either image files or loaded
            images.

    Returns:
        :obj:`DataSample` or list[:obj:`DataSample`]:
        If imgs is a list or tuple, the same length list type results
        will be returned, otherwise return the model results directly.
    """
    # prepare data
    data, is_batch = _preprare_data(img, model)

    # forward the model
    with torch.no_grad():
        results = model.test_step(data)

    return results if is_batch else results[0]


def visualize_result(cvlib,
                     visualizer,
                     img,
                     result,
                     pred_score_thr: float = 0.5,
                     title: str = '',
                     draw_gt: bool = True,
                     draw_pred: bool = True):
    """Visualize the results on the image.

    Args:
        visualizer: The loaded visualizer.
        img (str or np.ndarray): Image filename or loaded image.
        result (DataSample): The prediction DataSample result.
        pred_score_thr(float): bbox pred score thr.
            Default 0.5. Must be in (0, 1] range.
        title (str): The title of pyplot figure.
            Default is ''.
        draw_gt (bool): Whether to draw GT DataSample. Default to True.
        draw_pred (bool): Whether to draw Prediction DataSample.
            Defaults to True.

    Returns:
        np.ndarray: the drawn image which channel is BGR.
    """
    if isinstance(img, str):
        image = mmcv.imread(img, channel_order='rgb')
    else:
        image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    if "det" in cvlib or "yolo" in cvlib:
        visualizer.add_datasample(name=title,
                                  image=image,
                                  data_sample=result,
                                  draw_gt=draw_gt,
                                  draw_pred=draw_pred,
                                  wait_time=0,
                                  out_file=None,
                                  pred_score_thr=pred_score_thr)
    else:
        visualizer.add_datasample(name=title,
                                  image=image,
                                  data_sample=result,
                                  draw_gt=draw_gt,
                                  draw_pred=draw_pred,
                                  wait_time=0,
                                  out_file=None)
    vis_img = visualizer.get_image()
    drawn_image = cv2.cvtColor(vis_img, cv2.COLOR_RGB2BGR)

    return drawn_image