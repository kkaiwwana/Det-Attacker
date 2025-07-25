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


def disable_BN2d_track_running_stats(model: torch.nn.Module):
    """
    the variance and mean shifted while training adversarial patch/invisible global perturbetion.
    BatchNorm module track running stats in default setting, which may crash some models(like yolo_v3) in
    eval mode, especially when inffering on clean image, the means and vars are shifted to something
    that makes model can't work and even perform worse on clean image than image with adversarial samples.
    """
    for module in model.modules():
        if isinstance(module, torch.nn.BatchNorm2d):
            module.track_running_stats = False


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
    
    disable_BN2d_track_running_stats(my_model)
    
    return my_model


def fasterrcnn_resnet50_fpn_COCO():
    pretrained_model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
        weights=torchvision.models.detection.FasterRCNN_ResNet50_FPN_Weights.COCO_V1
    )
    out_channels = pretrained_model.backbone(torch.rand(1, 3, 1, 1))['0'].shape[1]
    my_model = FasterRCNN(DummyBackbone(out_channels), num_classes=91) # box_nms_thresh=0.2

    pretrained_weights = []
    for param in pretrained_model.parameters():
        pretrained_weights.append(param)

    my_model.backbone = pretrained_model.backbone
    for i, param in enumerate(my_model.parameters()):
        param.data = pretrained_weights[i].data
        param.requires_grad = False
    
    disable_BN2d_track_running_stats(my_model)
    
    return my_model


def _fasterrcnn_resnet50_fpn_Carla():
    # well, typical detection network works perfectly, efficiently here.
    pretrained_model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
        weights=torchvision.models.detection.FasterRCNN_ResNet50_FPN_Weights.COCO_V1
    )
    out_channels = pretrained_model.backbone(torch.rand(1, 3, 1, 1))['0'].shape[1]
    # we've actually known the rough shape of region, so we can actually improve anchor generation strategy
    

    defaults = {
        "min_size": 720,
        "max_size": 1080,
        "rpn_pre_nms_top_n_test": 100,
        "rpn_post_nms_top_n_test": 50,
        "rpn_pre_nms_top_n_train": 100,
        "rpn_post_nms_top_n_train": 50,
        "rpn_score_thresh": 0.05,
        "box_nms_thresh": 0.2
    }
    from torchvision.models.detection import FasterRCNN
    from torchvision.models.detection.anchor_utils import AnchorGenerator
    
    anchor_sizes = ((32,), (64,), (128,), (256,), (512,))
    aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes) 
    
    anchor_generator = AnchorGenerator(sizes=anchor_sizes, aspect_ratios=aspect_ratios)
    
    class DummyBackbone(torch.nn.Module):
        # to fool FasterRCNN class, because Faster RCNN requires a backbone network with attribute 'out_channels'
        def __init__(self, out_channels):
            super().__init__()
            self.out_channels = out_channels

    my_model = FasterRCNN(DummyBackbone(out_channels),
                          num_classes=5,
                          rpn_anchor_generator=anchor_generator,
                          **defaults
                          )
    # use backbone in pretrained faster rcnn model directly
    # skip complicated steps in building a backbone with FPN.
    my_model.backbone = pretrained_model.backbone
    
    disable_BN2d_track_running_stats(my_model)
            
    return my_model


class FasterRCNNShell(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
    
    def __call__(self, imgs, targets=None, toxic_targets=None):
        if self.training:
            loss_dict = self.model(imgs, targets)
            loss_dict['atk_loss_classifier'] = 0
            loss_dict['atk_loss_box_reg'] = 0
            loss_dict['atk_loss_objectness'] = 0
            loss_dict['atk_loss_rpn_box_reg'] = 0
            
            return loss_dict
        else:
            return self.model(imgs)


def fasterrcnn_resnet50_fpn_Carla(pretrained_model=None):
    if pretrained_model:
        return FasterRCNNShell(torch.load(pretrained_model))
    else:
        return FasterRCNNShell(_fasterrcnn_resnet50_fpn_Carla())

    
def obj_score_loss(outputs, target, model):
    
    loss = sum([(torch.nn.ReLU()(output[..., 4]).sum() / ((output[..., 4] > 0).sum() + 1).item()) for output in outputs])
    
    return 0, (0, loss, 0, 0)

    
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
            img_idx = torch.tensor([i], device=device).broadcast_to((target['boxes'].shape[0], 1))
            labels = target['labels'].unsqueeze(dim=1).cpu().apply_(
                    lambda x: self.labels_frcnn2yolo[x] if self.labels_frcnn2yolo else x).to(device)
            boxes = box_convert(target['boxes'], 'xyxy', 'cxcywh')
            
            trans_target = torch.concat([img_idx, labels, boxes], dim=1)
            trans_target[:, [-4, -2]] /= self.input_size[0]
            trans_target[:, [-3, -1]] /= self.input_size[1]

            yolo_targets.append(trans_target)

        return torch.cat(yolo_targets, dim=0).to(device)

    def _compute_loss(self, predictions, targets, toxic_targets=None) -> Dict[str, torch.Tensor or None]:
        _, loss_real_gt = obj_score_loss(
            predictions, targets, self.yolo_model) if targets is not None else (None, [None] * 4)
        # _.detach_()
        _, loss_toxic_gt = obj_score_loss(
            predictions, toxic_targets, self.yolo_model) if toxic_targets is not None else (None, [None] * 4)
        # _.detach_()
        
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
        if self.training:
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
    for param in yolo_v3_model.parameters():
        param.requires_grad = False
    
    disable_BN2d_track_running_stats(yolo_v3_model)
    
    return _FasterRCNN_Like_YOLO(yolo_v3_model, input_size, coco_annotation_path)