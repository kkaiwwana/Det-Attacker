ExpDirRoot = 'Exps/'
ExpName = '07-16-10-10/'
ConfigFilePath = 'config.py'

# Description
description = 'Conv Patch generator, loss in paper, new Projector'

# Seed
seed = 46

# device
device = 'cuda'

# Dataset
# sum of train dataset length + valid_ds_len should be less than 5,000
dataset = 'COCO2017_val'
dataset_size = 5000
folder_path = '../autodl-tmp/COCO_dev/val2017/'
annotation_path = '../autodl-tmp/COCO_dev/annotations/instances_val2017.json'
if_resize = True
target_size = (480, 480)
train_ds_size = 1
valid_ds_size = 100

# Patch
patch_type = 'TpConv'
patch_init = 'random'
patch_size = 48, 48
expand_stages = 2

# Projector
pattern_posi = (0, 0, 20, 20)
random_posi = False
pattern_scale = (0.7, 1.2)
rotation_angle = (-20, 20)
pattern_padding = 0
mix_rate = (0.6, 1.0)
color_brush = None
min_luminance = 0.0
luminance_smooth_boundary = None
style_converter = None

# Attack
suppress_cats = 'all'
suppress_area = None
import torch
extra_target = {'boxes': torch.tensor([[0, 0, 480, 480.0]]), 'labels': torch.tensor([2])}

# Loss
loss_weight = {
    'atk_loss_classifier': 0.0,
    'atk_loss_box_reg': 0.0,
    'atk_loss_objectness': - 0.0,
    'atk_loss_rpn_box_reg': - 0.0,
    'loss_classifier': - 1.0,
    'loss_box_reg': - 1.0,
    'loss_objectness': - 1.0,
    'loss_rpn_box_reg': - 1.0,
}

# Training
mode = 'detection'
optimizer = 'Adam'
lr = 0.01
weight_decay = None
lr_scheduler = None
batch_size = 1
num_epochs = 10
iters_per_image = 300
num_workers = 16

# Network
network = 'fasterrcnn_resnet50_fpn_COCO'
# network = 'fasterrcnn_mobilenet_v3_large_320_fpn_COCO'


# metric
test_clean_image = True
test_batch_size = 32

# others
num_imgs2show = 20
log_loss_after_iters = 10