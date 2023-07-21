import pytorchyolo.utils.loss
import torch
import torchvision
from FasterRCNN.faster_rcnn import FasterRCNN
from FasterRCNN.anchor_utils import AnchorGenerator
from utils.utils import ResizeGroundTruth
from typing import *


class DummyBackbone(torch.nn.Module):
    # to fool FasterRCNN class
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


class _FasterRCNN_Like_YOLO:
    # TODO: Implement YOLO in pytorch
    def __init__(self, yolo_model, input_size):
        self.yolo_model = yolo_model
        self.input_size = input_size

    def _yolo_image_trans(self, imgs):
        trans = torchvision.transforms.Resize(size=self.input_size, antialias=True)
        return torch.stack([trans(img) for img in imgs], dim=0)

    def _yolo_target_trans(self, targets):
        yolo_target = []
        resizer = ResizeGroundTruth(self.input_size)
        for target in targets:
            target = resizer(target)
        raise NotImplementedError()

    def _compute_loss(self) -> Dict[str, torch.Tensor]:
        raise NotImplementedError()

    def __call__(self, images, targets):
        images = self._yolo_image_trans(images)
        targets = self._yolo_target_trans(targets)

        raise NotImplementedError()

    def train(self):
        self.yolo_model.train()

    def eval(self):
        self.yolo_model.eval()

    def to(self, device):
        self.yolo_model.to(device)


def yolo_v3(model_cfg_path='models/detection/YOLOv3/config/yolov3.cfg',
            model_weight_path='models/detection/YOLOv3/weights/yolov3.weights',
            input_size=(416, 416)):

    from pytorchyolo import models
    yolo_v3_model = models.load_model(model_path=model_cfg_path, weights_path=model_weight_path)

    return _FasterRCNN_Like_YOLO(yolo_v3_model, input_size)