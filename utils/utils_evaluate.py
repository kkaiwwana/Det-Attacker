import torch
import numpy as np
import torchvision.transforms as transforms
import PIL.Image
from typing import Optional, List
from torch import Tensor
from torchvision.utils import draw_bounding_boxes
from torchvision import transforms
import matplotlib.pyplot as plt
import torchvision.transforms.functional as F
from torchmetrics.detection.mean_ap import MeanAveragePrecision
import torch.utils.data as data
from utils import default_collate_fn

class AdvDetectionMetrics:
    """
    Compute metrics on detection for adversarial patch

    """
    def __init__(self, model, pattern_projector, img_transforms):
        self.model = model
        self.projector = pattern_projector
        self.trans = img_transforms
        self.metrics = {}

    def _compute_mAP(self):

        raise NotImplementedError()

    def _compute_ABNI(self):
        # compute Average Box Number Increase in the patch area
        raise NotImplementedError()

    def _compute_ABNS(self):
        # compute Average Box Number Suppress outside the patch area
        raise NotImplementedError()

    @staticmethod
    def _labels_not_in_region(targets, device):
        # return idx of targets not in suppress region
        # self.suppress_region is not None
        if targets[0]['boxes'].shape[0] == 0:
            return torch.tensor([])
        not_suppress_idx = []
        _fl = torch.tensor([True, True, False, False], device=device)
        for target in targets:
            if len(target['boxes']) == 0:
                not_suppress_idx.append(torch.tensor([], device=device))
                continue
            not_in_region_idx = ((target['boxes'] >= torch.tensor([0, 0, 202, 202.0]).to(device)) != _fl).any(dim=1)

        return not_in_region_idx

    def compute(self, patch, dataset, test_clear_imgs=False, batch_size=16, device='cuda', num_workers=1):

        self.model.to(device)
        self.model.eval()
        dl = data.DataLoader(dataset, batch_size=batch_size, collate_fn=default_collate_fn, num_workers=num_workers)
        for imgs, targets in dl:
            imgs_with_patch = self.projector(imgs.clone(), patch.to(device))

            bbox_with_patch = self.model(imgs_with_patch)
            result_with_patch = self._labels_not_in_region(bbox_with_patch, device)
            self.metrics['num_boxes_with_patch'] = \
                (self.metrics['num_boxes_with_patch'] if 'num_boxes_with_patch' in self.metrics.keys() else 0) \
                + result_with_patch.shape[0]
            # todo something here
            # self.metrics['nu']

def simple_watcher(epoch, Y, Y_hat, valid_Y, valid_Y_hat, f):
    # on classification, compute ASR (Attack Success Rate)
    train_acc = (Y_hat.argmax(dim=1) != Y).sum().item() / Y_hat.shape[0]
    valid_acc = (valid_Y_hat.argmax(dim=1) != valid_Y).sum().item() / Y_hat.shape[0]
    log(f'epoch: {epoch} train ASR: {100 * train_acc:.1f}% valid ASR: {100 * valid_acc:.1f}%', f=f)


def log(*args, f=None):
    # simple log, print info to console and file
    print(args)
    if f:
        for item in args:
            f.write(str(item))
        f.write('\n')
