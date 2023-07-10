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


class DataVisualizer:
    """
    record data while training and visualize it
    """
    def __init__(self):
        self.data = {}

    def record(self, **kwargs):
        # pass kwargs(key and value pair) to record it, the new data will be appended to the data series.
        # e.g. Entity.record(**{'DATA_0': data_0, 'DATA_1': data_1})
        for key in kwargs.keys():
            if key in self.data.keys():
                self.data[key].append(kwargs[key])
            else:
                self.data[key] = [kwargs[key]]

    def visualize(self, x_axis_key: str, y_axis_keys: List[List or str], sub_figure_size=(7, 5)):
        """
        args:
            x_axis_key: the key of data you want to plot in x-axis(horizontal-axis), string
                it supports specifying one var in single plot so far, cuz that's all we need under this scenario.
            y_axis_keys: list of keys of vars you want to plot in y-axis(vertical-axis),
                you can GROUP some vars to plot them in one sub-figure for comparison or something.
        example:
            while(training):
                -> do something
                -> visualizer.record({
                    'EPOCH': epoch, 'LOSS': loss, 'Train_Accuracy': train_acc, 'Valid_Accuracy': valid_acc
                })
            -> end training
            -> visualizer.visualize(x_axis_key='EPOCH', y_axis_key=['LOSS', ['Train_Accuracy', 'Valid_Accuracy']])
            # in this way you'll see 2 subplots, which are loss-epoch fig and (train_acc, valid_acc)-epoch.
        """
        x = self.data[x_axis_key]
        sub_plots = len(y_axis_keys)
        plt.figure(figsize=(sub_figure_size[0], sub_figure_size[1] * sub_plots), frameon=True)
        for sub_plot_idx in range(sub_plots):
            plt.subplot(sub_plots, 1, sub_plot_idx + 1)
            if not isinstance(y_axis_keys[sub_plot_idx], list or tuple):
                y_axis_keys[sub_plot_idx] = [y_axis_keys[sub_plot_idx],]
            legend = []
            for curve_idx in range(len(y_axis_keys[sub_plot_idx])):
                plt.plot(x, self.data[y_axis_keys[sub_plot_idx][curve_idx]])
                legend.append(y_axis_keys[sub_plot_idx][curve_idx])
            plt.legend(legend, loc='upper left')
        plt.show()
        return plt

    def reset(self):
        self.data.clear()