from mmx.registry import MODELS
from mmyolo.models.dense_heads import YOLOv5HeadModule, YOLOv5Head, YOLOv7Head


@MODELS.register_module()
class DetYOLOv5HeadModule(YOLOv5HeadModule):
    """Inherit YOLOv5Head head module used in `YOLOv5` and 'YOLOv7'
    """

    def _forward(self, feats):
        """Forward features from the upstream network.

        Args:
            x (Tuple[Tensor]): Features from the upstream network, each is
                a 4D-tensor.
        Returns:
            Tuple[List]: A tuple of multi-level classification scores, bbox
            predictions, and objectnesses.
        """
        assert len(feats) == self.num_levels
        pred_maps = []
        for i in range(self.num_levels):
            x = feats[i]
            pred_map = self.convs_pred[i](x)
            pred_maps.append(pred_map)

        return tuple(pred_maps)


@MODELS.register_module()
class DetYOLOv5Head(YOLOv5Head):
    """Inherit YOLOv5Head head used in `YOLOv5`.
    """

    def _forward(self, x):
        """Forward features from the upstream network.

        Args:
            x (Tuple[Tensor]): Features from the upstream network, each is
                a 4D-tensor.
        Returns:
            Tuple[List]: A tuple of multi-level classification scores, bbox
            predictions, and objectnesses.
        """
        return self.head_module._forward(x)


@MODELS.register_module()
class DetYOLOv7Head(YOLOv7Head):
    """Inherit YOLOv7Head head used in `YOLOv7`.
    """

    def _forward(self, x):
        """Forward features from the upstream network.

        Args:
            x (Tuple[Tensor]): Features from the upstream network, each is
                a 4D-tensor.
        Returns:
            Tuple[List]: A tuple of multi-level classification scores, bbox
            predictions, and objectnesses.
        """
        return self.head_module._forward(x)