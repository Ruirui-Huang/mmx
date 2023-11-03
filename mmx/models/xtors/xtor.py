# Copyright (c) OpenMMLab. All rights reserved.
from mmyolo.models.detectors import YOLODetector as BaseYOLODetector
from mmseg.models.segmentors import EncoderDecoder as BaseEncoderDecoder
from mmx.registry import MODELS


@MODELS.register_module()
class YOLODetector(BaseYOLODetector):
    r"""Implementation of YOLO Series

    Args:
        backbone (:obj:`ConfigDict` or dict): The backbone config.
        neck (:obj:`ConfigDict` or dict): The neck config.
        bbox_head (:obj:`ConfigDict` or dict): The bbox head config.
        train_cfg (:obj:`ConfigDict` or dict, optional): The training config
            of YOLO. Defaults to None.
        test_cfg (:obj:`ConfigDict` or dict, optional): The testing config
            of YOLO. Defaults to None.
        data_preprocessor (:obj:`ConfigDict` or dict, optional): Config of
            :class:`DetDataPreprocessor` to process the input data.
            Defaults to None.
        init_cfg (:obj:`ConfigDict` or list[:obj:`ConfigDict`] or dict or
            list[dict], optional): Initialization config dict.
            Defaults to None.
        use_syncbn (bool): whether to use SyncBatchNorm. Defaults to True.
    """

    def forward_dummy(self, batch_inputs, batch_data_samples=None):
        """
        Args:
            batch_inputs (Tensor): Inputs with shape (N, C, H, W)
        Returns:
            tuple[list]: A tuple of features from "bbox_head" forward
        """
        x = self.extract_feat(batch_inputs)
        return self.bbox_head._forward(x)
        
@MODELS.register_module()
class EncoderDecoder(BaseEncoderDecoder):
     def forward_dummy(self, inputs):
        """
        Args:
            inputs (Tensor): Inputs with shape (N, C, H, W).
        Returns:
            Tensor: Forward output of model without any post-processes.
        """
        x = self.extract_feat(inputs)
        return self.decode_head.forward(x)

    def forward_particular(self, inputs):
        batch_img_metas = [
            dict(
                ori_shape=inputs.shape[2:],
                img_shape=inputs.shape[2:],
                pad_shape=inputs.shape[2:],
                padding_size=[0, 0, 0, 0]
            )
        ] * inputs.shape[0]
        return self.decode_head(inputs, batch_img_metas)
