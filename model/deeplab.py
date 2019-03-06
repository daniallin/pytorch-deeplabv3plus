import torch
import torch.nn as nn
import torch.nn.functional as F

from model.backbone import build_backbone
from model.aspp import ASPP
from model.decoder import Decoder


class DeepLabV3Plus(nn.Module):
    def __init__(self, backbone='resnst', sync_bn=False, num_classes=1000,
                 freeze_bn=False, output_scale=16):
        super(DeepLabV3Plus, self).__init__()
        if backbone == 'drn':
            output_scale = 8

        self.backbone = build_backbone(backbone, output_scale, sync_bn)
        self.aspp = ASPP(backbone, output_scale, sync_bn)
        self.decoder = Decoder(backbone, num_classes, sync_bn)

        if freeze_bn:
            for m in self.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()

    def forward(self, input):
        x, low_level_feature = self.backbone(input)
        x = self.aspp(x)
        x = self.decoder(x, low_level_feature)
        x = F.interpolate(x, size=input.size()[2:], mode='bilinear', align_corners=True)

        return x


if __name__ == '__main__':
    model = DeepLabV3Plus()
    model.eval()
    input = torch.rand(1, 3, 513, 513)
    output = model(input)
    print(output)

