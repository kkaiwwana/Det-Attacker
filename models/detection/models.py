import torch
import torchvision
from faster_rcnn.faster_rcnn import FasterRCNN
from faster_rcnn.anchor_utils import AnchorGenerator


class DummyBackbone(torch.nn.Module):
    # to fool faster_rcnn class
    def __init__(self, out_channels):
        super().__init__()
        self.out_channels = out_channels


def fasterrcnn_mobilenet_v3_large_320_fpn_COCO():
    # load official pretrained model to extract its weights
    # in this way, we can modify models and, meanwhile, get rid of training model by ourselves
    pretrained_model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_320_fpn(
        weights=torchvision.models.detection.FasterRCNN_MobileNet_V3_Large_320_FPN_Weights.COCO_V1
    )
    out_channels = pretrained_model.backbone(torch.rand(1, 3, 1, 1))['0'].shape[1]

    anchor_sizes = ((32, 64, 128, 256, 512,),) * 3
    aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)

    defaults = {
        "min_size": 320,
        "max_size": 640,
        "rpn_pre_nms_top_n_test": 150,
        "rpn_post_nms_top_n_test": 150,
        "rpn_score_thresh": 0.05,
    }

    my_model = FasterRCNN(DummyBackbone(out_channels),
                          num_classes=2,
                          rpn_anchor_generator=AnchorGenerator(anchor_sizes, aspect_ratios),
                          **defaults
                          )
    pretrained_weights = []
    for param in pretrained_model.parameters():
        pretrained_weights.append(param)

    my_model.backbone = pretrained_model.backbone
    for i, param in enumerate(my_model.parameters()):
        param.data = pretrained_weights[i].data
        param.requires_grad = False
    return my_model


def fasterrcnn_resnet50_fpn_COCO():
    pretrained_model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
        weights=torchvision.models.detection.FasterRCNN_ResNet50_FPN_Weights.COCO_V1
    )
    out_channels = pretrained_model.backbone(torch.rand(1, 3, 1, 1))['0'].shape[1]
    my_model = FasterRCNN(DummyBackbone(out_channels), num_classes=91)

    pretrained_weights = []
    for param in pretrained_model.parameters():
        pretrained_weights.append(param)

    my_model.backbone = pretrained_model.backbone
    for i, param in enumerate(my_model.parameters()):
        param.data = pretrained_weights[i].data
        param.requires_grad = False
    return my_model
