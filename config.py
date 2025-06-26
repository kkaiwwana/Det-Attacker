import torch
from data import BoxNumSelector, TrafficScenesSelector
################################################################################################
ExpDirRoot = 'Exps/'
ExpName = '09-17-22-17/'
ConfigFilePath = 'config.py'
################################################################################################
# Description
description = 'test_in_carla'
################################################################################################
# Seed
seed = 42
################################################################################################
# device
device = 'cuda'
################################################################################################
# Dataset
# sum of train dataset length + valid_ds_len should be less than 5,000
dataset_id = 1

if dataset_id == 0:
    dataset = 'COCO2017_val'
    dataset_size = 5000
    folder_path = '../autodl-tmp/val2017/'
    annotation_path = '../autodl-tmp/annotations/instances_val2017.json'
    
    if_resize = True
    target_size = (480, 480)
    train_ds_size = 3
    valid_ds_size = 100
    data_selectors = [None, None]

elif dataset_id == 1:
    dataset = 'CARLA_dataset'
    dataset_size = 1500
    
    if_resize = False
    train_ds_size = 10
    valid_ds_size = 200
################################################################################################
# Patch
patch_type = 'TpConv'
patch_init = 'random'
patch_size = 32, 32
expand_stages = 2
if_finetune = False
finetune_patch = './Exps/07-19-12-33/Data/patch_generator.pt'
################################################################################################
# Projector
specify_indices = True

pattern_posi = (0, 0, 20, 20)
random_posi = False
pattern_scale = (0.7, 1.2)
rotation_angle = (-15, 15)
pattern_padding = 0
mix_rate = (0.5, 0.8)
color_brush = None
min_luminance = 0.0
luminance_smooth_boundary = None
style_converter = None

enable_dynamic_prj_params = False
dynamic_prj_params = {
    'strategy': 'linear',
    'pattern_posi': {'increment': (-0.1, -0.1, 0.1, 0.1), 'end_value': (0, 0, 200, 200)},
    'pattern_scale': {'increment': (-0.0001, 0.0001), 'end_value': (0.8, 1.2)},
    'rotation_angle': {'increment': (-0.001, 0.001), 'end_value': (-15, 15)},
    'mix_rate': {'increment': (-0.001, 0.001), 'end_value': (0.5, 1.0)},
}
################################################################################################
# Attack
suppress_cats = 'all'
suppress_area = None
extra_target = {'boxes': torch.tensor([
    [0, 0, 10, 10.0],
]), 'labels': torch.tensor([20])}
################################################################################################
# Loss
# faster rcnn losses
loss_weight = {
    'atk_loss_classifier': 0.0,
    'atk_loss_box_reg': 0.0,
    'atk_loss_objectness': 0.0,
    'atk_loss_rpn_box_reg': 0.0,
    'loss_classifier':  - 1.0,
    'loss_box_reg':  - 1.0,
    'loss_objectness':  - 1.0,
    'loss_rpn_box_reg':  - 1.0,
}
# yolov3 losses
yolo_loss_weight = {
    'atk_loss_box_reg': 0.0,
    'atk_loss_objectness': 0.0,
    'atk_loss_classifier': 0.0,
    'loss_box_reg': 0,
    'loss_objectness': 5.0,
    'loss_classifier': 0,
}
################################################################################################
# Training
mode = 'detection'
optimizer = 'Adam'
lr = 0.05
momentum = 0.9
weight_decay = 1e-4
lr_scheduler = None
batch_size = 1
num_epochs = 20
iters_per_image = 60
num_workers = 16
################################################################################################
# Network
network_id = 4

if network_id == 1:
    network = 'fasterrcnn_resnet50_fpn_COCO'
elif network_id == 2:
    network = 'fasterrcnn_mobilenet_v3_large_320_fpn_COCO'
elif network_id == 3:
    network = 'yolov3'
    yolo_input_size = (416, 416)
    yolo_config_path = 'models/detection/YOLOv3/config/yolov3.cfg'
    yolo_weight_path = 'models/detection/YOLOv3/weights/yolov3.weights'
elif network_id == 4:
    network = 'fasterrcnn_resnet50_fpn_Carla'
################################################################################################
# metric
test_clean_image = True
test_batch_size = 32
################################################################################################
# others
num_imgs2show = 20
log_loss_after_iters = 10
################################################################################################