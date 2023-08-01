import sys
import os
import random
import torch
import config
import pprint
import torchvision.transforms as transforms
from pycocotools.coco import COCO
from shutil import copyfile


if __name__ == '__main__':
    import config as cfg

    sys.path.append('models/detection')
    sys.path.append('models/detection/FasterRCNN')
    sys.path.append('utils')

    from models.detection.models import *
    from utils.evaluate_utils import AdvDetectionMetrics, TrainWatcher
    from utils.visualize_utils import draw_bbox_with_tensor
    from utils.utils import PatternProjector, ToxicTargetsGenerator, log, LossManager, ResizeGroundTruth, set_seed
    from adv_patch_generator import *
    from data import coco_2017_dev_5k
    from trainer import AdvPatchTrainer

    set_seed(cfg.seed)

    exp_file_dir = cfg.ExpDirRoot + cfg.ExpName
    os.makedirs(exp_file_dir, exist_ok=False)
    os.makedirs(exp_file_dir + 'Data', exist_ok=False)
    os.makedirs(exp_file_dir + 'Figures', exist_ok=False)
    os.makedirs(exp_file_dir + 'Figures/Examples', exist_ok=False)
    copyfile(src=cfg.ConfigFilePath, dst=exp_file_dir + 'config.txt')

    f = open(cfg.ExpDirRoot + cfg.ExpName + 'description.txt', 'w')
    f.write(cfg.description)
    f.close()

    if cfg.if_resize:
        img_trans = transforms.Compose([transforms.Resize(cfg.target_size, antialias=True), transforms.ToTensor()])
        target_trans = ResizeGroundTruth(cfg.target_size)
    else:
        img_trans = transforms.ToTensor()
        target_trans = None

    if cfg.dataset == 'COCO2017_val':
        train_ds, valid_ds, _ = coco_2017_dev_5k(folder_path=cfg.folder_path,
                                                 annotation_path=cfg.annotation_path,
                                                 img_trans=img_trans,
                                                 target_trans=target_trans,
                                                 split_rate=[cfg.train_ds_size,
                                                             cfg.valid_ds_size,
                                                             cfg.dataset_size - cfg.train_ds_size - cfg.valid_ds_size],
                                                 # data remained doesn't require a selector, set None 
                                                 data_selectors=cfg.data_selectors + [None] if cfg.data_selectors else None)
    else:
        assert False, f'dataset \'{cfg.dataset}\' not found/Implemented.'

    if cfg.network == 'fasterrcnn_resnet50_fpn_COCO':
        net2attack = fasterrcnn_resnet50_fpn_COCO()
    elif cfg.network == 'fasterrcnn_mobilenet_v3_large_320_fpn_COCO':
        net2attack = fasterrcnn_mobilenet_v3_large_320_fpn_COCO()
    elif cfg.network == 'yolov3':
        net2attack = yolo_v3(model_cfg_path=cfg.yolo_config_path,
                             model_weight_path=cfg.yolo_weight_path,
                             coco_annotation_path=cfg.annotation_path,
                             input_size=cfg.yolo_input_size)
    else:
        assert False, f'model \'{cfg.network}\' not found/implemented.'

    if cfg.if_finetune and cfg.finetune_patch:
        patch = torch.load(cfg.finetune_patch)
        if isinstance(patch, torch.Tensor):
            if cfg.patch_type == 'NoiseLike':
                patch_generator = NoiseLikePatch(patch.shape[1], patch.shape[2], init_mode=cfg.patch_init)
            elif cfg.patch_type == 'Conv':
                patch_generator = ConvGenerator(patch.shape[1], patch.shape[2], init_mode=cfg.patch_init)
            patch_generator.adv_patch = torch.nn.Parameter(patch)
        else:
            patch_generator = patch
    elif cfg.patch_type == 'NoiseLike':
        patch_generator = NoiseLikePatch(cfg.patch_size[0], cfg.patch_size[1], init_mode=cfg.patch_init)
    elif cfg.patch_type == 'TpConv':
        patch_generator = TpConvGenerator(cfg.patch_size[0], cfg.patch_size[1], expand_stages=cfg.expand_stages)
    else:
        assert False, f'patch generator not found/load.'

    projector = PatternProjector(pattern_posi=cfg.pattern_posi,
                                 random_posi=cfg.random_posi,
                                 pattern_scale=cfg.pattern_scale,
                                 rotation_angle=cfg.rotation_angle,
                                 pattern_padding=cfg.pattern_padding,
                                 mix_rate=cfg.mix_rate,
                                 color_brush=cfg.color_brush,
                                 min_luminance=cfg.min_luminance,
                                 luminance_smooth_boundary=cfg.luminance_smooth_boundary,
                                 style_converter=cfg.style_converter,
                                 dynamic_prj_params=cfg.dynamic_prj_params if cfg.enable_dynamic_prj_params else None)

    if cfg.network != 'yolov3':
        loss_func = LossManager(log_loss_after_iters=cfg.log_loss_after_iters, **cfg.loss_weight)
    else:
        loss_func = LossManager(log_loss_after_iters=cfg.log_loss_after_iters, **cfg.yolo_loss_weight)

    # set None, Modify it if you need scheduler here.
    lr_scheduler = cfg.lr_scheduler

    if cfg.optimizer == 'Adam':
        optimizer = torch.optim.Adam(
            params=patch_generator.parameters(),
            lr=cfg.lr,
            weight_decay=cfg.weight_decay if cfg.weight_decay else 0.0
        )
    elif cfg.optimizer == 'SGD':
        optimizer = torch.optim.SGD(
            params=patch_generator.parameters(),
            lr=cfg.lr,
            momentum=cfg.momentum,
            weight_decay=cfg.weight_decay if cfg.weight_decay else 0.0
        )
    elif cfg.optimizer == 'FGSM':
        from utils.utils import FGSM
        optimizer = FGSM(
            params=patch_generator.parameters(),
            lr=cfg.lr,
            weight_central_decay=config.weight_decay if cfg.weight_decay else None
        )
    else:
        assert False, f'optimizer {cfg.optimizer} not found/implemented.'

    targets_generator = ToxicTargetsGenerator(suppress_cats=cfg.suppress_cats,
                                              suppress_area=cfg.suppress_area,
                                              extra_target=cfg.extra_target)
    trainer = AdvPatchTrainer(net2attack=net2attack,
                              patch_generator=patch_generator,
                              projector=projector,
                              loss_function=loss_func,
                              optimizer=optimizer,
                              scheduler=lr_scheduler,
                              targets_generator=targets_generator,
                              device=cfg.device)
    watcher = TrainWatcher()
    trainer.train(mode=cfg.mode,
                  train_ds=train_ds,
                  batch_size=cfg.batch_size,
                  num_epochs=cfg.num_epochs,
                  iters_per_image=cfg.iters_per_image,
                  valid_ds=valid_ds,
                  log_filepath=exp_file_dir + 'train_log.txt',
                  num_workers=cfg.num_workers,
                  train_watcher=watcher)

    torch.save(patch_generator, exp_file_dir + 'Data' + '/patch_generator.pt')
    torch.save(patch_generator(), exp_file_dir + 'Data' + '/patch.pt')
    watcher.save_data(filepath=exp_file_dir + 'Data', filename='/train_data.pt')
    watcher.save_fig(filepath=exp_file_dir + 'Figures', filename='/loss_curve.svg')

    train_metric = AdvDetectionMetrics(net2attack, projector, patch_generator)
    train_metric.compute(train_ds, test_clear_imgs=cfg.test_clean_image, batch_size=cfg.test_batch_size)
    valid_metric = AdvDetectionMetrics(net2attack, projector, patch_generator)
    valid_metric.compute(valid_ds, test_clear_imgs=cfg.test_clean_image, batch_size=cfg.test_batch_size)

    f = open(exp_file_dir + 'evaluation.txt', 'w')
    f.write(f'train:\n{pprint.pformat(train_metric.metrics)}\n\n'
            f'valid:\n{pprint.pformat(valid_metric.metrics)}')
    f.close()
    torch.save(train_metric.metrics, exp_file_dir + 'Data/train_metrics_dict.pt')
    torch.save(valid_metric.metrics, exp_file_dir + 'Data/valid_metrics_dict.pt')
    
    net2attack.eval()
    cats = COCO(cfg.annotation_path).cats

    def get_str_labels(int_labels):
        str_labels = []
        if isinstance(int_labels, int):
            int_labels = [int_labels]
        for label in int_labels:
            str_labels.append(cats[int(label)]['name'])
        return str_labels

    idx2show = random.randint(0, len(train_ds) + len(valid_ds) - cfg.num_imgs2show)
    
    for i in range(cfg.num_imgs2show):
        if idx2show + i < len(train_ds):
            img = train_ds[idx2show + i][0].to(config.device)
        else:
            img = valid_ds[idx2show + i - len(train_ds)][0].to(config.device)
        
        # check channels
        if img.shape[0] == 1:
            img = img.broadcast_to(3, -1, -1)
            
        preds_clean_img = net2attack((img.clone(),))
        boxes_clean_img = preds_clean_img[0]['boxes']
        int_labels_clean_image = preds_clean_img[0]['labels']
        str_labels_clean_image = get_str_labels(int_labels_clean_image)
        example_image_clean = draw_bbox_with_tensor(img=img.clone(), bbox=boxes_clean_img, label=str_labels_clean_image)

        img_with_patch = projector(img.clone(), patch_generator().to(cfg.device))[0]
        preds_with_patch = net2attack((img_with_patch,))
        boxes_with_patch = preds_with_patch[0]['boxes']
        int_labels_with_patch = preds_with_patch[0]['labels']
        str_labels_with_patch = get_str_labels(int_labels_with_patch)
        example_with_patch = draw_bbox_with_tensor(img=img_with_patch, bbox=boxes_with_patch, label=str_labels_with_patch)
        
        example_image = torchvision.transforms.ToPILImage()(
            torch.concat((example_image_clean, example_with_patch), dim=1)
        )
        example_image.save(f'{exp_file_dir}Figures/Examples/{i + 1}.jpg')