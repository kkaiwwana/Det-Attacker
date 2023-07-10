ExpDirRoot = 'Exps/'
ExpName = '7-10-14-17/'
ConfigFilePath = 'config.py'

# Description
description = 'This Is A Test.'

# Seed
seed = 42

# device
device = 'cuda'

# Dataset
# sum of train dataset length + valid_ds_len should be less than 5,000
dataset = 'COCO2017_val'
dataset_size = 5000
folder_path = './datasets/COCO_dev/val2017/'
annotation_path = './datasets/COCO_dev/annotations/instances_val2017.json'
if_resize = True
target_size = (480, 480)
train_ds_size = 10
valid_ds_size = 20

# Patch
patch_type = 'NoiseLike'
patch_init = 'random'
patch_size = 128, 128

# Projector
pattern_posi = (0, 0)
random_posi = False
pattern_scale = 1.0
pattern_padding = 0
mix_rate = 0.9
color_brush = None
min_luminance = None
luminance_smooth_boundary = None
style_converter = None

# Attack
suppress_cats = 'all'
suppress_area = None
extra_target = 'global'

# Loss
loss_weight = {
    'atk_loss_classifier': 0.0,
    'atk_loss_box_reg': 0.0,
    'atk_loss_objectness': - 1.0,
    'atk_loss_rpn_box_reg': 0.0,
    'loss_classifier': - 0.0,
    'loss_box_reg': - 0.0,
    'loss_objectness': - 0.0,
    'loss_rpn_box_reg': - 0.0,
}

# Training
mode = 'detection'
optimizer = 'Adam'
lr = 0.1
weight_decay = None
lr_scheduler = None
batch_size = 2
num_epochs = 5
iters_per_image = 3
num_workers = 1

# Network
network = 'fasterrcnn_resnet50_fpn_COCO'

# metric
test_clean_image = True
test_batch_size = 2

# others
num_imgs2show = 10