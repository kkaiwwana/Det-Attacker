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


class AdvDetectionMetrics:
    """
    Compute metrics on detection for adversarial patch

    """
    def __init__(self):
        raise NotImplementedError()

    def _compute_mAP(self):

        raise NotImplementedError()

    def _compute_ABNI(self):
        # compute Average Box Number Increase in the patch area
        raise NotImplementedError()

    def _compute_ABNS(self):
        # compute Average Box Number Suppress outside the patch area
        raise NotImplementedError()

    def compute(self):
        raise NotImplementedError()


def simple_watcher(epoch, Y, Y_hat, valid_Y, valid_Y_hat, f):
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
