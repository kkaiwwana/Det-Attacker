import torch
from data import BoxNumSelector, TrafficScenesSelector

ExpDirRoot = 'Exps/'
ExpName = '07-27-12-46/'
ConfigFilePath = 'config.py'

# Description
description = 'yolo'

# Seed
seed = 42

# device
device = 'cuda'

# Dataset
# sum of train dataset length + valid_ds_len should be less than 5,000
dataset = 'COCO2017_val'
dataset_size = 5000
folder_path = '../autodl-tmp/val2017/'
annotation_path = '../autodl-tmp/annotations/instances_val2017.json'
if_resize = True
target_size = (480, 480)
train_ds_size = 5
valid_ds_size = 100
data_selectors = [TrafficScenesSelector(8), TrafficScenesSelector(8)]

# Patch

patch_type = 'NoiseLike'
patch_init = 'random'
patch_size = 32, 32
expand_stages = 2
if_finetune = False
finetune_patch = 'Exps/07-19-12-33/Data/patch_generator.pt'

# Projector
pattern_posi = (0, 0, 0, 0)
random_posi = False
pattern_scale = (1.0, 1.0)
rotation_angle = (-0, 0)
pattern_padding = 0
mix_rate = (0.1, 0.1)
color_brush = None
min_luminance = 0.0
luminance_smooth_boundary = None
style_converter = None

enable_dynamic_prj_params = False
dynamic_prj_params = {
    'strategy': 'linear',
    'pattern_posi': {'increment': (-0.1, -0.1, 0.1, 0.1), 'end_value': (0, 0, 200, 200)},
    'pattern_scale': {'increment': (-0.01, 0.01), 'end_value': (0.5, 1.6)},
    'rotation_angle': {'increment': (-0.01, 0.01), 'end_value': (-35, 35)},
    'mix_rate': {'increment': (-0.01, 0.01), 'end_value': (0.5, 1.0)},
}

# Attack
suppress_cats = 'all'
suppress_area = None
extra_target = {'boxes': torch.tensor([[0, 0, 0.1, 0.1]]), 'labels': torch.tensor([2])}

# Loss
# faster rcnn losses
loss_weight = {
    'atk_loss_classifier': 0.0,
    'atk_loss_box_reg': 0.0,
    'atk_loss_objectness': 0.0,
    'atk_loss_rpn_box_reg': 0.0,
    'loss_classifier': - 1.0,
    'loss_box_reg': - 1.0,
    'loss_objectness': - 1.0,
    'loss_rpn_box_reg': - 1.0,
}
# yolov3 losses
yolo_loss_weight = {
    'atk_loss_box_reg': 0.0,
    'atk_loss_objectness': - 10.0,
    'atk_loss_classifier': 0.0,
    'loss_box_reg': -0.0,
    'loss_objectness': -0.0,
    'loss_classifier': -0.0,
}

# Training
mode = 'detection'
optimizer = 'SGD'
lr = 0.1
momentum = 0.9
weight_decay = None
lr_scheduler = None
batch_size = 1
num_epochs = 10
iters_per_image = 30
num_workers = 16

# Network
network_id = 3

if network_id == 1:
    network = 'fasterrcnn_resnet50_fpn_COCO'
elif network_id == 2:
    network = 'fasterrcnn_mobilenet_v3_large_320_fpn_COCO'
elif network_id == 3:
    network = 'yolov3'
    yolo_input_size = (416, 416)
    yolo_config_path = 'models/detection/YOLOv3/config/yolov3.cfg'
    yolo_weight_path = 'models/detection/YOLOv3/weights/yolov3.weights'

# metric
test_clean_image = True
test_batch_size = 32

# others
num_imgs2show = 20
log_loss_after_iters = 10