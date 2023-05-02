import torch
import torch.nn as nn
from torchvision.models import resnet50
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from torchvision.ops.misc import FrozenBatchNorm2d


class ResNet(nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()

        for parent in resnet50().named_children():
            # if parent[0] != 'fc':
            self.__setattr__(parent[0], parent[1])

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return x

    def init_weight(self):
        faster_rcnn_resnet50 = fasterrcnn_resnet50_fpn(
            weights=FasterRCNN_ResNet50_FPN_Weights.COCO_V1
        )
        resnet_weights = faster_rcnn_resnet50.backbone.body

        bn_to_replace = []
        for name, module in resnet_weights.named_modules():
            if isinstance(module, FrozenBatchNorm2d):
                print('adding ', name)
                bn_to_replace.append(name)

        for layer_name in bn_to_replace:
            *parent, child = layer_name.split('.')
            if len(parent) > 0:
                m = resnet_weights.__getattr__(parent[0])
                for p in parent[1:]:
                    m = m.__getattr__(p)
                original_layer = m.__getattr__(child)
            else:
                m = resnet_weights.__getattr__(child)
                original_layer = m
            in_channels = original_layer.weight.shape[0]
            bn = nn.BatchNorm2d(in_channels)
            with torch.no_grad():
                bn.weight = nn.Parameter(original_layer.weight)
                bn.bias = nn.Parameter(original_layer.bias)
                bn.running_mean = original_layer.running_mean
                bn.running_var = original_layer.running_var
            m.__setattr__(child, bn)

        resnet_weights.bn1 = resnet_weights.bn1.bn1

        self.load_state_dict(resnet_weights.state_dict())
