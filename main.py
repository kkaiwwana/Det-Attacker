import sys
import torch
from adv_patch_generator import NoiseLikePatch
from data import coco_2017_dev_5k
from trainer import AdvPatchTrainer
from utils.utils import PatternProjector, ToxicTargetsGenerator
from utils.utils_evaluate import log
sys.path.append('models/detection/faster_rcnn')


class LossFunction:
    def __init__(self, target_class=0, alpha=0.3, beta=1.5):
        self.target_class = target_class
        self.alpha, self.beta = alpha, beta
        self.loss_func = torch.nn.CrossEntropyLoss()

    def __call__(self, y_hat, y):
        y_target = torch.ones_like(y) * self.target_class
        return - self.alpha * self.loss_func(y_hat, y) + self.beta * self.loss_func(y_hat, y_target)


if __name__ == '__main__':
    from models.detection.faster_rcnn.my_pretrained_faster_rcnn import fasterrcnn_mobilenet_v3_large_320_fpn_COCO

    # print('WATCH OUT, it\'s ATTACK!')
    # train_ds, valid_ds = imagenet_1k_mini()
    # net2attack = resnet_18()
    # patch_generator = NoiseLikePatch(H_size=64, W_size=64)
    # projector = PatternProjector(mix_rate=0.0)
    # loss_func = LossFunction()
    #
    # trainer = AdvPatchTrainer(net2attack=net2attack,
    #                           patch_generator=patch_generator,
    #                           projector=projector,
    #                           loss_function=loss_func,
    #                           optimizer=torch.optim.Adam(patch_generator.parameters(), lr=0.002))
    # trainer.train('classification',
    #               train_ds,
    #               batch_size=4,
    #               num_epochs=2,
    #               valid_ds=valid_ds,
    #               log_filepath='./5-10-log.txt',
    #               num_workers=8,
    #               train_watcher=simple_watcher)

    # model = fasterrcnn_mobilenet_v3_large_320_fpn_COCO()
    # # model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_320_fpn(
    # #     weight=torchvision.models.detection.FasterRCNN_MobileNet_V3_Large_320_FPN_Weights.COCO_V1
    # # )
    # # model.to('cuda')
    # DS = coco_2017_dev_5k()
    #
    #
    # def collate_fn(batch):
    #     return tuple(zip(*batch))
    #
    #
    # DL = torch.utils.data.DataLoader(DS, batch_size=8, shuffle=True, collate_fn=collate_fn)
    #
    # patch_generator = NoiseLikePatch(H_size=96, W_size=96)
    # projector = PatternProjector(mix_rate=0.99, luminance_smooth_boundary=0.01, min_luminance=0)
    # optimizer = torch.optim.Adam(patch_generator.parameters(), lr=0.1)
    # TTG = ToxicTargetsGenerator()
    # for _ in range(1):
    #     optimizer.zero_grad()
    #     mean_loss = 0
    #     for idx, (X, y) in enumerate(DL):
    #         skip = False
    #         for item in y:
    #             if len(item['boxes'].shape) == 1:
    #                 skip = True
    #                 break
    #
    #         X = list(X)
    #         pattern = patch_generator()
    #         for i in range(len(X)):
    #             X[i] = X[i].broadcast_to(3, X[i].shape[-2], X[i].shape[-1]).clone()
    #             if X[i].shape[1] < 96 or X[i].shape[2] < 96:
    #                 skip = True
    #                 break
    #
    #             X[i] = projector.project_pattern(X[i], pattern)[0]
    #             # X[i] = X[i].to('cuda')
    #
    #         if skip:
    #             continue
    #
    #         for ann in y:
    #             # ann['boxes'] = ann['boxes'].to('cuda')
    #             ann['boxes'][:, 2: 4] += 0.1
    #             # ann['labels'] = ann['labels'].to('cuda')
    #         toxic = TTG(y, 'cpu')
    #         loss_dict = model(X, y)
    #         loss = (loss_dict['atk_loss_classifier'] + loss_dict['atk_loss_box_reg'] +
    #                 loss_dict['atk_loss_objectness'] + loss_dict['atk_loss_rpn_box_reg']) * 2.0 - \
    #                (loss_dict['loss_classifier'] + loss_dict['loss_box_reg'] +
    #                 loss_dict['loss_objectness'] + loss_dict['loss_rpn_box_reg']) * 0.5
    #
    #         mean_loss += loss.item()
    #         if (idx + 1) % 30 == 0:
    #             print(mean_loss / 30)
    #             mean_loss = 0
    #
    #         loss.backward()
    #
    #         optimizer.step()

    train_ds, valid_ds = coco_2017_dev_5k(split_rate=[0.8, 0.2])
    net2attack = fasterrcnn_mobilenet_v3_large_320_fpn_COCO()
    patch_generator = NoiseLikePatch(96, 96)
    projector = PatternProjector(mix_rate=0.99, min_luminance=0.01, luminance_smooth_boundary=0.01)

    def loss_manager(loss_dict, f):
        loss = (loss_dict['atk_loss_classifier'] + loss_dict['atk_loss_box_reg'] +
                loss_dict['atk_loss_objectness'] + loss_dict['atk_loss_rpn_box_reg']) * 2.0 - \
               (loss_dict['loss_classifier'] + loss_dict['loss_box_reg'] +
                loss_dict['loss_objectness'] + loss_dict['loss_rpn_box_reg']) * 0.5
        log(loss.item(), f=f)
        return loss

    loss_func = loss_manager
    optimizer = torch.optim.SGD(patch_generator.parameters(), lr=0.1, momentum=0.9)
    targets_generator = ToxicTargetsGenerator()
    trainer = AdvPatchTrainer(net2attack,
                              patch_generator,
                              projector,
                              loss_func,
                              optimizer,
                              targets_generator=targets_generator)

    def watcher(data, f):
        log(data, f)

    trainer.train('detection', train_ds, 8, 2, valid_ds, "./detect_log_5_22.txt", 1, train_watcher=watcher)
