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


def draw_bbox_with_tensor(img: torch.Tensor, bbox: torch.Tensor, label=None):
    # draw_bounding_boxes implemented in torchvision supports image input with dtype=uint8 only
    # that's too annoying. so, build this wheel to relief my pain :)
    return draw_bounding_boxes(transforms.PILToTensor()(transforms.ToPILImage()(img)), bbox, labels=label)


def show(imgs):
    if not isinstance(imgs, list):
        imgs = [imgs]
    fig, axs = plt.subplots(ncols=len(imgs), squeeze=False)
    for i, img in enumerate(imgs):
        img = img.detach()
        img = F.to_pil_image(img)
        axs[0, i].imshow(np.asarray(img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

