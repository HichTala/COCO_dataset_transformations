import logging
from typing import List, Optional, Dict

import torch
from torch import nn
from detectron2.modeling.poolers import ROIPooler
from detectron2.structures import ImageList, Instances
from detectron2.modeling import build_backbone

from bottleneckblock_wo_relu import BottleneckBlock

logger = logging.getLogger(__name__)


class ResNetROIPool(nn.Module):

    def __init__(self, cfg):
        super(ResNetROIPool, self).__init__()

        self.device = torch.device(cfg.MODEL.DEVICE)

        self.pixel_mean = torch.Tensor(cfg.MODEL.PIXEL_MEAN).to(self.device).view(3, 1, 1)
        self.pixel_std = torch.Tensor(cfg.MODEL.PIXEL_STD).to(self.device).view(3, 1, 1)

        self.backbone = build_backbone(cfg)
        block = BottleneckBlock(1024, 1024,
                                stride=1,
                                norm='FrozenBN',
                                bottleneck_channels=256,
                                stride_in_1x1=True,
                                dilation=1,
                                num_groups=1)
        self.backbone.res4[5] = block

        self.in_features = cfg.MODEL.ROI_HEADS.IN_FEATURES
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))

        input_shape = self.backbone.output_shape()
        pooler_resolution = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        pooler_type = cfg.MODEL.ROI_BOX_HEAD.POOLER_TYPE
        pooler_scales = (1.0 / input_shape[self.in_features[0]].stride,)
        sampling_ratio = cfg.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO

        self.pooler = ROIPooler(
            output_size=pooler_resolution,
            scales=pooler_scales,
            sampling_ratio=sampling_ratio,
            pooler_type=pooler_type,
        )

    def forward(self, batched_inputs):

        images = self.preprocess_image(batched_inputs)
        targets = [x["instances"].to(self.device) for x in batched_inputs]

        features = self.backbone(images.tensor)

        target_classes = [x.gt_classes for x in targets]
        target_boxes = [x.gt_boxes for x in targets]

        box_features = self.pooler(
            [features[f] for f in self.in_features], target_boxes
        )
        box_features = self.avgpool(box_features).squeeze(dim=2).squeeze(dim=2)

        return box_features, target_classes

    def preprocess_image(self, batched_inputs):
        """
        Normalize, pad and batch the input images.
        """
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = ImageList.from_tensors(
            images,
            self.backbone.size_divisibility,
            padding_constraints=self.backbone.padding_constraints,
        )

        return images
