import torch

from mmx.registry import MODELS
from mmcls.models import GlobalAveragePooling


@MODELS.register_module()
class ClsGAP(GlobalAveragePooling):

    def forward(self, inputs):
        if isinstance(inputs, tuple):
            outs = tuple([self.gap(x) for x in inputs])
            outs = tuple([torch.flatten(out, 1) for out, x in zip(outs, inputs)])
        elif isinstance(inputs, torch.Tensor):
            outs = self.gap(inputs)
            outs = torch.flatten(outs, 1)
        else:
            raise TypeError('neck inputs should be tuple or torch.tensor')
        return outs