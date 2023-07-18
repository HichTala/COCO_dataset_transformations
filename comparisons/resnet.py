import torch
import torch.nn as nn
from detectron2.modeling import build_backbone
from detectron2.structures import ImageList

from bottleneckblock_wo_relu import BottleneckBlock


class ResNet(nn.Module):
    def __init__(self, cfg):
        super(ResNet, self).__init__()

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
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))

    def forward(self, batched_inputs):
        images = self.preprocess_image(batched_inputs)

        features = self.backbone(images.tensor)
        features = [features[f] for f in features.keys()]
        features = torch.cat(features)
        features = self.avgpool(features)

        features = torch.flatten(features, 1)
        return features

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
