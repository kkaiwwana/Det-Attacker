import operator

import torch
import numpy as np
import PIL.Image
import os
import random
from typing import *
from torch import Tensor
from torchvision import transforms
from evaluate_utils import log


class ResizeGroundTruth:
    def __init__(self, target_size):
        self.target_size = target_size

    def __call__(self, target):
        device = target['boxes'].device
        origin_size = target['origin_size'][-1]
        sc_rate = self.target_size[0] / origin_size[0], self.target_size[1] / origin_size[1]
        target['boxes'] *= torch.tensor([sc_rate[0], sc_rate[1], sc_rate[0], sc_rate[1]], device=device)
        target['origin_size'].append(torch.tensor(self.target_size))

        return target


class LossManager:
    def __init__(self, log_loss_after_iters=10, **loss_weight_dict):
        self.weight_dict = loss_weight_dict
        self.log_loss_after_iters = log_loss_after_iters
        self.iters_count = 0
        self.mean_loss = 0
        self.is_training = True

    def __call__(self, loss_dict: Dict[str, Tensor or None], log_file=None):
        weighted_loss = {}
        for key in self.weight_dict.keys():
            if key in loss_dict.keys() and loss_dict[key] is not None:
                weighted_loss[key] = self.weight_dict[key] * loss_dict[key]

        loss = sum(weighted_loss.values())

        if self.is_training is False:
            return loss

        self.mean_loss += loss.item()
        self.iters_count += 1
        if (self.iters_count + 1) % self.log_loss_after_iters == 0:
            log(f'Total Iters: {self.iters_count + 1} '
                f'Loss: {round(self.mean_loss / self.log_loss_after_iters, 3)}', f=log_file)
            self.mean_loss = 0

        return loss

    def train(self):
        self.is_training = True

    def eval(self):
        self.is_training = False


def default_collate_fn(batch):
    return tuple(zip(*batch))


class ToxicTargetsGenerator:
    def __init__(self,
                 suppress_cats: List[int] or Tensor or str = None,
                 suppress_area: List[int or float] or Tensor = None,
                 extra_target: Dict[str, Tensor] or str = None
                 ):
        """
        generate 'toxic' labels from real annotation(boxes, labels) as supervise signal to train adv pattern
        Args:
            suppress_cats: categories of detected objects you want to suppress, default None means not suppress.
                you can use string 'all' or 'none' to specify your purpose.
            suppress_area: region of detected objects you want to suppress, default None means suppress GLOBAL

        """
        if suppress_cats:
            if isinstance(suppress_cats, str) or isinstance(suppress_cats, Tensor):
                self.suppress_cats = suppress_cats
            elif isinstance(suppress_cats, list):
                self.suppress_cats = torch.tensor(suppress_cats)
        else:
            self.suppress_cats = torch.tensor([])
        if suppress_area:
            self.suppress_area = suppress_area if isinstance(suppress_area, Tensor) else torch.tensor(suppress_area)
        else:
            self.suppress_area = torch.tensor([0, 0, torch.inf, torch.inf])

        self.extra_target = extra_target

    def _labels_not_in_area(self, targets, device) -> List[Tensor]:
        # return idx of targets not in suppress region
        # self.suppress_area is not None
        not_suppress_idx = []
        _fl = torch.tensor([True, True, False, False], device=device)
        for target in targets:
            if len(target['boxes']) == 0:
                not_suppress_idx.append(torch.tensor([], device=device))
                continue
            not_in_region_idx = ((target['boxes'] >= self.suppress_area.to(device)) != _fl).any(dim=1)
            if isinstance(self.suppress_cats, str):
                if self.suppress_cats == 'all':
                    not_suppress_label = torch.zeros_like(target['labels'])
                elif self.suppress_cats == 'none':
                    not_suppress_label = torch.ones_like(target['labels'])
                else:
                    assert False, 'Wrong Suppress Mode Specified.'
            elif len(self.suppress_cats) != 0:
                not_suppress_label = torch.ones_like(target['labels'])
                for i in range(len(target['labels'])):
                    if target['labels'][i] in self.suppress_cats:
                        not_suppress_label[i] = 0
            else:
                not_suppress_label = torch.ones_like(target['labels'])
            not_suppress_idx.append(((not_suppress_label > 0) | not_in_region_idx).nonzero().squeeze(dim=-1))
        return not_suppress_idx

    def transform_targets(self, targets, device='cuda'):
        toxic_targets = []
        not_in_region_idx = self._labels_not_in_area(targets, device)

        for idx, target in zip(not_in_region_idx, targets):

            if len(idx) > 0:
                d = {'boxes': target['boxes'][idx].clone(), 
                     'labels': target['labels'][idx].clone(), 
                     'image_id': target['image_id'].clone()}
                
                if self.extra_target:
                    if isinstance(self.extra_target, str):
                        if self.extra_target == 'global':
                            global_box = torch.tensor([[0, 0, target['origin_size'][-1][0].clone(), target['origin_size'][-1][1].clone()]])
                            d['boxes'] = torch.concat((d['boxes'], global_box.to(device)), dim=0)
                            d['labels'] = torch.concat((d['labels'], torch.tensor([0], device=device)), dim=0)
                        else:
                            assert False, 'error.'
                    else:
                        d['boxes'] = torch.concat((d['boxes'], self.extra_target['boxes'].clone().to(device)), dim=0)
                        d['labels'] = torch.concat((d['labels'], self.extra_target['labels'].clone().to(device)), dim=0)
            else:
                if not self.extra_target:
                    d = {'boxes': torch.tensor([0, 0, 1e-2, 1e-2], device=device),
                         'labels': torch.tensor([0], device=device),
                         'image_id': target['image_id'].clone()}
                else:
                    if isinstance(self.extra_target, str):
                        global_box = torch.tensor([[0, 0, target['origin_size'][-1][0].clone(), target['origin_size'][-1][1].clone()]])
                        d = {'boxes': global_box.to(device),
                             'labels': torch.tensor([0], device=device),
                             'image_id': target['image_id'].clone(),
                             'origin_size': list(size.clone() for size in target['origin_size'])}
                    else:
                        d = {'boxes': self.extra_target['boxes'].clone().to(device),
                             'labels': self.extra_target['labels'].clone().to(device),
                             'image_id': target['image_id'].clone(),
                             'origin_size': list(size.clone() for size in target['origin_size'])}

            toxic_targets.append(d)

        return toxic_targets

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
                 pattern_posi: Tuple or str = (0, 0),
                 specify_indices: bool = False,
                 pattern_scale: Tuple[int or float] or float = 1.0,
                 rotation_angle: Tuple[int or float] or float = None,
                 pattern_padding: int = 0,
                 mix_rate: Tuple[float, float] or float = 1.0,
                 color_brush: int or Tuple[float, float, float] = None,
                 min_luminance: float = 0.05,
                 luminance_smooth_boundary=0.6,
                 style_converter=None,
                 dynamic_prj_params=None,
                 **kwargs):
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
            dynamic_prj_params: a dict with items: 'strategy': '[MODE]'(required)， '[PARAM NAME]': '[Args]'
                e.g. :
                    d_p_dict = {
                        'strategy': 'linear',
                        'pattern_posi': {'increment': (0.2, -0.2, 0.1, 0.1), 'end_value': (10, 10, 20, 20)},
                        'pattern_scale': {'increment': (-0.1, 0.2), 'end_value': (0.2, 1.8)}
                    }
        """
        self.random_posi = False
        if isinstance(pattern_posi, str):
            if pattern_posi == 'random':
                pattern_posi = (0, 0, 0, 0)
                self.random_posi = True
            else:
                assert False, f'unknown argument pattern_pois={pattern_posi}'

        pattern_posi = pattern_posi if len(pattern_posi) == 4 else pattern_posi * 2
        pattern_scale = pattern_scale if isinstance(pattern_scale, (tuple, list)) else (pattern_scale,) * 2
        rotation_angle = rotation_angle if isinstance(rotation_angle, (tuple, list)) else (rotation_angle,) * 2
        pattern_padding = pattern_padding
        mix_rate = mix_rate if isinstance(mix_rate, (tuple, list)) else (mix_rate,) * 2
        color_brush = color_brush
        min_luminance = min_luminance
        luminance_smooth_boundary = luminance_smooth_boundary
        self.prj_params = {
            'pattern_posi': pattern_posi,
            'pattern_scale': pattern_scale,
            'rotation_angle': rotation_angle,
            'pattern_padding': pattern_padding,
            'mix_rate': mix_rate,
            'color_brush': color_brush,
            'min_luminance': min_luminance,
            'luminance_smooth_boundary': luminance_smooth_boundary
        }
        self.style_converter = style_converter

        self.dynamic_prj_params = dynamic_prj_params
        self._iter_counter = 0

        # arg random posi was removed, to support old version code, keep this.
        self.random_posi = self.random_posi or (kwargs['random_posi'] if 'random_posi' in kwargs else False)
        
        self.specify_indices = specify_indices

    def _dynamic_prj_params(self):
        def _modify_value(op):
            if op == 'mul':
                op = operator.mul
            elif op == 'add':
                op = operator.add

            for param_name, increment in self.dynamic_prj_params.items():
                if isinstance(increment, dict):
                    if 'end_value' not in self.dynamic_prj_params[param_name].keys():
                        self.prj_params[param_name] = \
                            tuple([v + i for v, i in zip(self.prj_params[param_name],
                                                         self.dynamic_prj_params[param_name]['increment'])])
                    else:
                        self.prj_params[param_name] = \
                            tuple([min(op(v, i), e, key=lambda x: abs(x)) for v, i, e in
                                   zip(self.prj_params[param_name],
                                       self.dynamic_prj_params[param_name]['increment'],
                                       self.dynamic_prj_params[param_name]['end_value'])])

        self._iter_counter += 1
        if self.dynamic_prj_params['strategy'] == 'linear':
            _modify_value(op='add')

        elif self.dynamic_prj_params['strategy'] == 'stepwise':
            _steps = self.dynamic_prj_params['steps']
            if self._iter_counter % _steps == 0:
                _modify_value(op='add')

        elif self.dynamic_prj_params['strategy'] == 'exponential':
            _modify_value(op='mul')

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
        if isinstance(self.prj_params['color_brush'], tuple):
            if isinstance(self.prj_params['color_brush'][0], int):
                # convert int RGB to (0, 1)
                R = self.prj_params['color_brush'][0] / 255.0
                G = self.prj_params['color_brush'][1] / 255.0
                B = self.prj_params['color_brush'][2] / 255.0
            else:
                R, G, B = self.prj_params['color_brush'][0], \
                    self.prj_params['color_brush'][1], \
                    self.prj_params['color_brush'][2]
        else:
            # convert wave length to RGB
            R, G, B = PatternProjector._wavelength_to_rgb(self.prj_params['color_brush'])
        
        device = img_tensor.device

        if len(img_tensor.shape) == 3:
            img_tensor = (~mask) * torch.tensor([[[R]], [[G]], [[B]]], device=device).broadcast_to(
                (3, img_tensor.shape[1], img_tensor.shape[2]))
        else:
            img_tensor = (~mask) * torch.tensor([[[[R]], [[G]], [[B]]]], device=device).broadcast_to(
                (-1, 3, img_tensor.shape[2], img_tensor.shape[3]))

        return img_tensor

    def _luminance_smooth_mask(self, luminance):
        norm_lumi = normalize_tensor(luminance)
        _mask = (norm_lumi > self.prj_params['luminance_smooth_boundary'])
        weighted_mask = (_mask + (~_mask) * luminance / self.prj_params['luminance_smooth_boundary'] + 1e-5)
        return weighted_mask

    def project_pattern(self, img, pattern, patch_indices=None):
        """
        Parameters:
            img: object image, support image filepath/3-D tensor/4-D tensor(a batch)
            pattern: your pattern
            
        Returns:
            image_tensor: image tensor with input pattern injected in
            pattern: input pattern, it will be broadcast to a 4-D tensor when input image(s) is a 4-D tensor
        """
        if self.specify_indices:
            def _patch_downsample(img, target_size):
                return transforms.Resize(target_size)(img)
             
            patch_tensor = _patch_downsample(pattern, patch_indices.shape[2:])
            for i in range(patch_indices.shape[0]):
                mix_rate = random.uniform(*self.prj_params['mix_rate'])
                img[:, patch_indices[i][0], patch_indices[i][1]] = img[:, patch_indices[i][0], patch_indices[i][1]] * (1 - mix_rate) + patch_tensor * mix_rate
            
            return img, patch_tensor, (patch_indices.shape[-2], patch_indices.shape[-1])
                
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

        # padding
        pattern_tensor = torch.nn.ConstantPad2d(value=0, padding=self.prj_params['pattern_padding'])(pattern_tensor)

        # random rotation
        if self.prj_params['rotation_angle'][0]:
            pattern_tensor = transforms.RandomRotation(self.prj_params['rotation_angle'], expand=True)(pattern_tensor)

        # random scale
        scale = random.uniform(*self.prj_params['pattern_scale'])
        pattern_H = int(pattern_tensor.shape[-2] * scale)
        pattern_W = int(pattern_tensor.shape[-1] * scale)
        pattern_tensor = transforms.Resize((pattern_H, pattern_W))(pattern_tensor)

        # convert style
        if self.style_converter:
            pattern_tensor = self.style_converter(pattern_tensor)

        # operation on luminance、mask、weighted_mask are NOT differentiable
        # so, you should detach these stuffs.
        luminance = PatternProjector._rgb2luminance(pattern_tensor).detach()
        mask = luminance <= self.prj_params['min_luminance'] if self.prj_params['min_luminance'] is not None \
            else torch.zeros_like(luminance) > 0

        mix_rate = random.uniform(*self.prj_params['mix_rate'])

        if self.prj_params['luminance_smooth_boundary']:
            weighted_mask = self._luminance_smooth_mask(luminance) * mix_rate
        else:
            weighted_mask = torch.ones_like(luminance) * mix_rate

        if self.prj_params['color_brush']:
            pattern_tensor = self._color_brush(pattern_tensor, mask)

        _max_H, _max_W = img_H - pattern_H, img_W - pattern_W
        if not self.random_posi:
            if (torch.tensor([self.prj_params['pattern_posi']]) >= 1).any():
                # pixel position, e.g. (123, 456), top left corner
                posi_x = max(0, min(random.randrange(
                    int(self.prj_params['pattern_posi'][0]), int(self.prj_params['pattern_posi'][2]) + 1), _max_H))
                posi_y = max(0, min(random.randrange(
                    int(self.prj_params['pattern_posi'][1]), int(self.prj_params['pattern_posi'][3]) + 1), _max_W))
            else:
                # relative position, e.g. (0.3, 0.4), top left corner
                posi_x = max(0, min(int(img_H * random.uniform(
                    self.prj_params['pattern_posi'][0], self.prj_params['pattern_posi'][2])), _max_H))
                posi_y = max(0, min(int(img_W * random.uniform(
                    self.prj_params['pattern_posi'][1], self.prj_params['pattern_posi'][3])), _max_W))
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
        
        if self.dynamic_prj_params:
            self._dynamic_prj_params()
            
        
        return img_tensor, pattern_tensor * weighted_mask, (posi_x, posi_y)

    def __call__(self, img, pattern, patch_indices=None):
        return self.project_pattern(img, pattern, patch_indices)

    
class Projector:
    def __init__(self, transparency=None):
        self.transparency = transparency
    
    @staticmethod
    def _patch_downsample(img, target_size):
        return transforms.Resize(target_size)(img)
        
    def __call__(self, img, patch_data, patch_indices):
        patch_data = Projector._patch_downsample(self.patch_data, patch_indices.shape[2:])
        
        for i in range(patch_indices.shape[0]):
            img[:, patch_indices[i][0], patch_indices[i][1]] = patch_data
        
        # TODO: compute metric for new projector
        return img, None, (patch_indices[0][0], patch_indices[0][1])


def set_seed(seed: int = 42) -> None:
    if seed:
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)
        # torch.Generator().manual_seed(seed)
        torch.cuda.manual_seed(seed)
        # When running on the CuDNN backend, two further options must be set
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # Set a fixed value for the hash seed
        os.environ["PYTHONHASHSEED"] = str(seed)
        print(f"Random seed set as {seed}")


class FGSM:
    def __init__(self, params, lr, weight_central_decay=None):
        self.params = params
        self.lr = lr
        self.weight_central_decay = weight_central_decay

    def step(self):
        with torch.no_grad():
            for param in self.params:
                param -= self.lr * torch.sign(param.grad)
                if self.weight_central_decay:
                    param *= (1 - self.weight_central_decay) ** torch.sign(param - 0.5)

    def zero_grad(self):
        for param in self.params:
            if param.grad is not None:
                param.grad.data.fill_(0.0)
