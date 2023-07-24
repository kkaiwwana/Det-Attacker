import torch
import torchvision
import torchvision.transforms as transforms
from typing import *
from torchvision.ops.boxes import box_convert
from FasterRCNN.faster_rcnn import FasterRCNN
from FasterRCNN.anchor_utils import AnchorGenerator
from utils.utils import ResizeGroundTruth
from pycocotools import coco
import YOLOv3.models as models
from YOLOv3.utils.loss import compute_loss
from YOLOv3.detect import detect_image


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


class _FasterRCNN_Like_YOLO(torch.nn.Module):

    def __init__(self, yolo_model, input_size, coco_annotation_path=None):
        super().__init__()
        self.yolo_model = yolo_model
        self.input_size = input_size
        if coco_annotation_path:
            cats = coco.COCO(coco_annotation_path).cats
            self.labels_frcnn2yolo = {key: i for i, key in enumerate(cats.keys())}
            self.labels_yolo2frcnn = {i: key for i, key in enumerate(cats.keys())}
        else:
            self.labels_frcnn2yolo, self.labels_yolo2frcnn = None, None

    def _yolo_image_trans(self, imgs):
        trans = torchvision.transforms.Resize(size=self.input_size, antialias=True)
        return torch.stack([trans(img) for img in imgs], dim=0).to(imgs[0].device)

    def _yolo_target_trans(self, targets: Tuple[Dict[str, torch.Tensor]]):
        device = targets[0]['boxes'].device
        yolo_targets = []
        resizer = ResizeGroundTruth(self.input_size)
        for i, target in enumerate(targets):
            target = resizer(target)
            trans_target = torch.concat([
                torch.tensor([i], device=device, requires_grad=False).broadcast_to((target['boxes'].shape[0], 1)),
                target['labels'].unsqueeze(dim=1).cpu().apply_(
                    lambda x: self.labels_frcnn2yolo[x] if self.labels_frcnn2yolo else x).to(device).to(torch.int),
                box_convert(target['boxes'], 'xyxy', 'cxcywh')
            ], dim=1)
            trans_target[:, [-4, -2]] /= self.input_size[0]
            trans_target[:, [-3, -1]] /= self.input_size[1]

            yolo_targets.append(trans_target)

        return torch.cat(yolo_targets, dim=0).to(device)

    def _compute_loss(self, predictions, targets, toxic_targets=None) -> Dict[str, torch.Tensor or None]:
        _, loss_real_gt = compute_loss(
            predictions, targets, self.yolo_model) if targets is not None else (None, [None] * 4)
        _, loss_toxic_gt = compute_loss(
            predictions, toxic_targets, self.yolo_model) if toxic_targets is not None else (None, [None] * 4)

        losses_dict = {
            'loss_box_reg': loss_real_gt[0],
            'loss_objectness': loss_real_gt[1],
            'loss_classifier': loss_real_gt[2],
            'atk_loss_box_reg': loss_toxic_gt[0],
            'atk_loss_objectness': loss_toxic_gt[1],
            'atk_loss_classifier': loss_toxic_gt[2],
        }
        return losses_dict

    def __call__(self, images, targets=None, toxic_targets=None):
        device = images[0].device
        if self.yolo_model.training:
            yolo_images = self._yolo_image_trans(images)
            yolo_targets = self._yolo_target_trans(targets) if targets else None
            yolo_toxic_targets = self._yolo_target_trans(toxic_targets) if toxic_targets else None

            preds = self.yolo_model(yolo_images)

            losses_dict = self._compute_loss(preds, yolo_targets, yolo_toxic_targets)
            return losses_dict
        else:
            detects = [torch.tensor(
                detect_image(
                    model=self.yolo_model,
                    image=(img * 256).to(torch.uint8).permute(1, 2, 0).cpu().numpy(),
                    img_size=self.input_size[0]),
                device=device
            ) for img in images]
            detects = tuple(
                {'boxes': detect[:, 0: 4],
                 'labels': detect[:, 5].cpu().apply_(lambda x: self.labels_yolo2frcnn[x] if self.labels_yolo2frcnn else x).to(device).to(torch.int),
                 'scores': detect[:, 4]} for detect in detects)
            return detects


def yolo_v3(model_cfg_path='models/detection/YOLOv3/config/yolov3.cfg',
            model_weight_path='models/detection/YOLOv3/weights/yolov3.weights',
            coco_annotation_path='datasets/COCO_dev/annotations/instances_val2017.json',
            input_size=(416, 416)):

    yolo_v3_model = models.load_model(model_path=model_cfg_path, weights_path=model_weight_path)

    return _FasterRCNN_Like_YOLO(yolo_v3_model, input_size, coco_annotation_path)