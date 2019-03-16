from model.backbone import resnet, xception, inception3
# from model.backbone import resnet, xception, drn, mobilenet


def build_backbone(backbone, output_scale, sync_bn=False, pretrained=False):
    if backbone == 'resnet':
        return resnet.resnet101(pretrained, output_scale=output_scale, sync_bn=sync_bn)
    elif backbone == 'inception':
        return inception3.inception_v3(pretrained, sync_bn=sync_bn)

    # elif backbone == 'xception':
    #     return xception.AlignedXception(output_scale, sync_bn=sync_bn)
    # elif backbone == 'drn':
    #     return drn.drn_d_54(sync_bn=sync_bn)
    # elif backbone == 'mobilenet':
    #     return mobilenet.MobileNetV2(output_stride, sync_bn=sync_bn)
    else:
        raise NotImplementedError



