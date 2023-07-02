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


def default_collate_fn(batch):
    return tuple(zip(*batch))


class ToxicTargetsGenerator:
    def __init__(self,
                 suppress_cats: List[int] or Tensor = None,
                 suppress_region: List[int or float] or Tensor = None,
                 generate_cats=None,
                 generate_region=None,
                 ):
        """
        generate 'toxic' labels from real annotation(boxes, labels) as supervise signal to train adv pattern
        Args:
            suppress_cats: categories of detected objects you want to suppress, default None means suppress ALL
            suppress_region: region of detected objects you want to suppress, default None means suppress GLOBAL
            generate_cats: TODO
            generate_region: TODO
        """
        if suppress_cats:
            self.suppress_cats = suppress_cats if isinstance(suppress_cats, Tensor) else torch.tensor(suppress_cats)
        else:
            self.suppress_cats = torch.tensor([])
        if suppress_region:
            self.suppress_region = suppress_region if isinstance(suppress_region, Tensor) else torch.tensor(
                suppress_region)
        else:
            self.suppress_region = torch.tensor([0, 0, torch.inf, torch.inf])
        self.generate_cats = generate_cats
        self.generate_region = generate_region

    def _labels_not_in_region(self, targets, device) -> List[Tensor]:
        # return idx of targets not in suppress region
        # self.suppress_region is not None
        not_suppress_idx = []
        _fl = torch.tensor([True, True, False, False], device=device)
        for target in targets:
            if len(target['boxes']) == 0:
                not_suppress_idx.append(torch.tensor([], device=device))
                continue
            not_in_region_idx = ((target['boxes'] >= self.suppress_region.to(device)) != _fl).any(dim=1)
            if len(self.suppress_cats) != 0:
                not_suppress_label = torch.ones_like(target['labels'])
                for i in range(len(target['labels'])):
                    if target['labels'][i] in self.suppress_cats:
                        not_suppress_label[i] = 0
            else:
                not_suppress_label = torch.zeros_like(target['labels'])
            not_suppress_idx.append(((not_suppress_label > 0) | not_in_region_idx).nonzero().squeeze(dim=-1))
        return not_suppress_idx

    def transform_targets(self, targets, device='cuda'):
        toxic_targets = []
        not_in_region_idx = self._labels_not_in_region(targets, device)
        for idx, target in zip(not_in_region_idx, targets):
            d = {'boxes': target['boxes'][idx] if len(idx) > 0 else torch.tensor([0, 0, 1.0, 1.0], device=device),
                 'labels': target['labels'][idx] if len(idx) > 0 else torch.tensor([0], device=device),
                 'image_id': target['image_id']}
            toxic_targets.append(d)

        return tuple(toxic_targets)

    def __call__(self, *args, **kwargs):
        return self.transform_targets(*args, **kwargs)


def normalize_tensor(x, max_value=1.0, min_value=0.0):
    # c_max = x.max(dim=-1, keepdim=True)[0].max(dim=-2, keepdim=True)[0]
    # c_min = x.min(dim=-1, keepdim=True)[0].min(dim=-2, keepdim=True)[0]
    if len(x.shape) == 3:
        c_max, c_min = x.max(), x.min()
    elif len(x.shape) == 4:
        c_max, c_min = x.amax(dim=(-3, -2, -1), keepdim=True), x.amin(dim=(-3, -2, -1), keepdim=True)
    else:
        return x
    return (x - c_min) / (c_max - c_min) * (max_value - min_value) + min_value


class LaserStyleConverter:
    def __init__(self, radius: int = 10, sigma: float = 0.1, depth: int = 1, device='cpu'):
        self.radius = radius
        self.depth = depth
        self.sigma = sigma
        self.device = device

        self.conv_tp = self._get_convTp()

    def _get_convTp(self):
        k = self.radius * 2 + 1
        weight = torch.nn.Parameter(torch.zeros((3, 1, k, k), device=self.device), requires_grad=False)
        for x in range(k):
            for y in range(k):
                weight[:, :, x, y] = np.exp(- self.sigma ** 2 * ((x - self.radius) ** 2 + (y - self.radius) ** 2))
        conv_tp = \
            torch.nn.ConvTranspose2d(3, 3, kernel_size=(k, k), stride=1, padding=self.radius, groups=3, bias=False)

        conv_tp._parameters['weight'] = weight
        return conv_tp

    def to(self, device):
        self.conv_tp.to(device)

    @staticmethod
    def _add_laserEdge2_pattern(laser_edge, pattern):
        return torch.stack((laser_edge, pattern), dim=0).max(dim=0)[0]

    def convert(self, img_tensor):
        X = normalize_tensor(self.conv_tp(img_tensor))
        for _ in range(self.depth - 1):
            X = normalize_tensor(self.conv_tp(X))
        return LaserStyleConverter._add_laserEdge2_pattern(img_tensor, X)

    def __call__(self, arg):
        return self.convert(arg)


class PatternProjector:
    def __init__(self,
                 pattern_posi: (int, int) or (float, float) = (0, 0),
                 random_posi=False,
                 pattern_scale: float = 1.0,
                 pattern_padding=0,
                 mix_rate: float = 0.8,
                 color_brush: int or (float, float, float) = None,
                 min_luminance: float = 0.05,
                 luminance_smooth_boundary=0.6,
                 style_converter=None):
        """
        Description:
        Project some pattern to an image

        Parameters:
            pattern_posi:
                the position of pattern in the object image. support absolute(pixel) position and relative position
            random_posi: project pattern to random position. param: pattern_posi will be ignored when it's enabled
            pattern_scale: scale pattern's size, range from 0 to some positive scalar.
                ERROR OCCURS when scaling pattern larger than input image.
            pattern_padding: pad input pattern with 0, support single value and tuple
            mix_rate: transparency, range in [0, 1],
                '0' means pattern is invisible, '1' means pixels on object image are totally covered
            color_brush: convert pattern to single color,
                support (R, G, B) input, where R, G, B are integers range in (0, 255) or real numbers range in [0, 1]
                and wave_length input, range in (380nm, 750nm)
            min_luminance: the boundary that decide if a pixel will be projected to the image.
                a pixel will be projected to the object image if its value is over min_luminance, otherwise, it won't.
            luminance_smooth_boundary: smooth the edge or parts with light luminance of pattern, make it looks natural.
            style_converter: a callable and differentiable object that convert image input to expected style.
                you can customize one to convert your pattern or something to something you wanted.
                ATTENTION, its ‘__call__‘ method MUST BE implemented.
        """
        self.pattern_posi = pattern_posi
        self.random_posi = random_posi
        self.pattern_scale = pattern_scale
        self.pattern_padding = pattern_padding
        self.mix_rate = mix_rate
        self.color_brush = color_brush
        self.min_luminance = min_luminance
        self.luminance_smooth_boundary = luminance_smooth_boundary
        self.style_converter = style_converter

    @staticmethod
    def _rgb2luminance(img_tensor):
        """
        Convert RGB to luminance can be calculated from linear RGB components:
        Y = 0.2126 * R + 0.7152 * G + 0.0722 * B
        See more details in wiki: https://en.wikipedia.org/wiki/Relative_luminance
        """
        if len(img_tensor.shape) == 3:
            # a torch img_tensor, shape: (C: [R, G, B], H, W)
            return img_tensor[0] * 0.2126 + img_tensor[1] * 0.7152 + img_tensor[2] * 0.0722
        elif len(img_tensor.shape) == 4:
            # img_tensor in batch, shape: (batch, C: [R, G, B], H, W)
            return img_tensor[0, 0] * 0.2126 + img_tensor[0, 1] * 0.7152 + img_tensor[0, 2] * 0.0722
        else:
            return img_tensor

    @staticmethod
    def _wavelength_to_rgb(wavelength, gamma=0.8):
        """
        Description:
        Given a wavelength in the range of (380nm, 750nm), visible light range.
        a tuple of integers for (R,G,B) is returned.
        The integers are scaled to the range (0, 1).

        Based on code: https://www.noah.org/wiki/Wavelength_to_RGB_in_Python
        Parameters:
            wavelength: the given wavelength range in (380, 750)
        Returns:
            (R,G,B): color range in (0,1)
        """
        wavelength = float(wavelength)
        R, G, B = 0.0, 0.0, 0.0
        if 380 <= wavelength <= 440:
            attenuation = 0.3 + 0.7 * (wavelength - 380) / (440 - 380)
            R, G, B = ((-(wavelength - 440) / (440 - 380)) * attenuation) ** gamma, 0.0, (1.0 * attenuation) ** gamma
        elif 440 <= wavelength <= 490:
            R, G, B = 0.0, ((wavelength - 440) / (490 - 440)) ** gamma, 1.0
        elif 490 <= wavelength <= 510:
            R, G, B = 0.0, 1.0, (-(wavelength - 510) / (510 - 490)) ** gamma
        elif 510 <= wavelength <= 580:
            R, G, B = ((wavelength - 510) / (580 - 510)) ** gamma, 1.0, 0.0
        elif 580 <= wavelength <= 645:
            R, G, B = 1.0, (-(wavelength - 645) / (645 - 580)) ** gamma, 0.0
        elif 645 <= wavelength <= 750:
            attenuation = 0.3 + 0.7 * (750 - wavelength) / (750 - 645)
            R, G, B = (1.0 * attenuation) ** gamma, 0.0, 0.0
        return R, G, B

    def _color_brush(self, img_tensor, mask):
        R, G, B = 0.0, 0.0, 0.0
        if isinstance(self.color_brush, tuple):
            if isinstance(self.color_brush[0], int):
                # convert int RGB to (0, 1)
                R = self.color_brush[0] / 255.0
                G = self.color_brush[1] / 255.0
                B = self.color_brush[2] / 255.0
            else:
                R, G, B = self.color_brush[0], self.color_brush[1], self.color_brush[2]
        else:
            # convert wave length to RGB
            R, G, B = PatternProjector._wavelength_to_rgb(self.color_brush)

        if len(img_tensor.shape) == 3:
            img_tensor = (~mask) * torch.tensor([[[R]], [[G]], [[B]]]).broadcast_to(
                (3, img_tensor.shape[1], img_tensor.shape[2]))
        else:
            img_tensor = (~mask) * torch.tensor([[[[R]], [[G]], [[B]]]]).broadcast_to(
                (-1, 3, img_tensor.shape[2], img_tensor.shape[3]))

        return img_tensor

    def _luminance_smooth_mask(self, luminance):
        norm_lumi = normalize_tensor(luminance)
        _mask = (norm_lumi > self.luminance_smooth_boundary)
        weighted_mask = (_mask + (~_mask) * luminance / self.luminance_smooth_boundary + 1e-5) * self.mix_rate
        return weighted_mask

    def project_pattern(self, img, pattern):
        """
        Parameters:
            img: object image, support image filepath/3-D tensor/4-D tensor(a batch)
            pattern: your pattern
        Returns:
            image_tensor: image tensor with input pattern injected in
            pattern: input pattern, it will be broadcast to a 4-D tensor when input image(s) is a 4-D tensor
        """
        img_tensor, pattern_tensor = img, pattern
        batch_size = None
        # when img is a file path, open it and convert it to a tensor
        if isinstance(img, str):
            img_tensor = transforms.ToTensor()(PIL.Image.open(img))
        if isinstance(pattern, str):
            pattern_tensor = transforms.ToTensor()(PIL.Image.open(pattern))
        if len(img_tensor.shape) == 4:
            batch_size = img_tensor.shape[0]
            img_H, img_W = img_tensor.shape[2], img_tensor.shape[3]
        else:
            img_H, img_W = img_tensor.shape[1], img_tensor.shape[2]

        pattern_tensor = torch.nn.ConstantPad2d(value=0, padding=self.pattern_padding)(pattern_tensor)
        pattern_H = int(pattern_tensor.shape[-2] * self.pattern_scale)
        pattern_W = int(pattern_tensor.shape[-1] * self.pattern_scale)

        pattern_tensor = transforms.Resize((pattern_H, pattern_W))(pattern_tensor)
        if self.style_converter:
            pattern_tensor = self.style_converter(pattern_tensor)

        # operation on luminance、mask、weighted_mask are NOT differentiable
        # so, you should detach these stuffs.
        luminance = PatternProjector._rgb2luminance(pattern_tensor).detach()
        mask = luminance < self.min_luminance if self.min_luminance else torch.zeros_like(luminance) > 0
        if self.luminance_smooth_boundary:
            weighted_mask = self._luminance_smooth_mask(luminance)
        else:
            weighted_mask = torch.ones_like(luminance)

        if self.color_brush:
            pattern_tensor = self._color_brush(pattern_tensor, mask)

        _max_H, _max_W = img_H - pattern_H, img_W - pattern_W
        if not self.random_posi:
            if isinstance(self.pattern_posi[0], int):
                # pixel position, e.g. (123, 456), top left corner
                posi_x = min(self.pattern_posi[0], _max_H)
                posi_y = min(self.pattern_posi[1], _max_W)
            else:
                # relative position, e.g. (0.3, 0.4), top left corner
                posi_x = min(int(img_H * self.pattern_posi[0]), _max_H)
                posi_y = min(int(img_W * self.pattern_posi[1]), _max_W)
        else:
            posi_x, posi_y = torch.randint(0, _max_H, (1,)), torch.randint(0, _max_W, (1,))

        if not batch_size:
            img_patch = img_tensor[:, posi_x: posi_x + pattern_H, posi_y: posi_y + pattern_W]
            img_tensor[:, posi_x: posi_x + pattern_H, posi_y: posi_y + pattern_W] = \
                img_patch * mask + (~mask) * (img_patch * (1 - weighted_mask) + pattern_tensor * weighted_mask)
        else:
            if len(pattern_tensor.shape) == 3:
                pattern_tensor = pattern_tensor.squeeze(dim=0).broadcast_to((batch_size,) + pattern_tensor.shape)
            img_patch = img_tensor[:, :, posi_x: posi_x + pattern_H, posi_y: posi_y + pattern_W]
            img_tensor[:, :, posi_x: posi_x + pattern_H, posi_y: posi_y + pattern_W] = \
                img_patch * mask + (~mask) * (img_patch * (1 - weighted_mask) + pattern_tensor * weighted_mask)
        return img_tensor, pattern_tensor * weighted_mask

    def __call__(self, img, pattern):
        return self.project_pattern(img, pattern)