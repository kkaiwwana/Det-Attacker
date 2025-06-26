import torch
import torch.nn as nn
from utils.utils import normalize_tensor


class NoiseLikePatch(torch.nn.Module):
    def __init__(self, H_size, W_size, init_mode='random'):
        super().__init__()
        self.patch_H, self.patch_W = H_size, W_size
        if init_mode == 'random':
            self.adv_patch = nn.Parameter(torch.rand((3, H_size, W_size), requires_grad=True))
        elif init_mode == 'all_zeros':
            self.adv_patch = nn.Parameter(torch.zeros((3, H_size, W_size), requires_grad=True))
        elif init_mode == 'all_ones':
            self.adv_patch = nn.Parameter(torch.ones((3, H_size, W_size), requires_grad=True))
        else:
            assert False, 'unknown init mode passed.'
    
    def forward(self, batch_size=None):
        return self.adv_patch.clamp(0.0001, 1)


class TpConvGenerator(nn.Module):
    def __init__(self, H_init, W_init, expand_stages=2):
        super().__init__()
        self.patch_H, self.patch_W = H_init, W_init
        self.latent_matrix = nn.Parameter(torch.normal(0, 0.5 / 6, size=(3, H_init, W_init), requires_grad=True))
        layers = []
        for _ in range(expand_stages):
            layers.append(torch.nn.ConvTranspose2d(3, 3, kernel_size=2, stride=2))

        layers.append(torch.nn.Conv2d(3, 32, kernel_size=5, padding=2))
        layers.append(torch.nn.Conv2d(32, 3, kernel_size=5, padding=2))
        layers.append(nn.BatchNorm2d(3))

        self.cal_seq = nn.Sequential(*layers)

    def forward(self, batch_size=None):
        return (self.cal_seq(self.latent_matrix.unsqueeze(dim=0))[0] + 0.5).clamp(0.0001, 1)
    

class ConvGenerator(nn.Module):
    def __init__(self, H_init, W_init, init_mode='random'):
        super().__init__()
        self.patch_H, self.patch_W = H_init, W_init
        self.adv_patch = nn.Parameter(torch.normal(0, 0.5 / 6, size=(3, H_init, W_init), requires_grad=True))
        layers = []

        layers.append(torch.nn.Conv2d(3, 32, kernel_size=5, padding=2))
        # layers.append(torch.nn.Conv2d(32, 32, kernel_size=5, padding=2))
        layers.append(torch.nn.Conv2d(32, 3, kernel_size=5, padding=2))
        layers.append(nn.BatchNorm2d(3))

        self.cal_seq = nn.Sequential(*layers)

    def forward(self, batch_size=None):
        return (self.cal_seq(self.adv_patch.unsqueeze(dim=0))[0] + 0.5).clamp(0.0001, 1)


class PrintableGenerator(nn.Module):

    def __init__(self, H_init, W_init, colors='8bit'):
        super().__init__()
        self.patch_H, self.patch_W = H_init, W_init
        self.latent_matrix = nn.Parameter(torch.normal(0, 0.5 / 6, size=(3, H_init, W_init), requires_grad=True))
        if colors == '8bit':
            self.colors = torch.arange(1, 256) / 255.0

        layers = []
        layers.append(torch.nn.ConvTranspose2d(3, 32, kernel_size=2, stride=2))
        layers.append(torch.nn.Conv2d(32, 32, kernel_size=5, padding=2))
        layers.append(torch.nn.ConvTranspose2d(32, 3, kernel_size=2, stride=2))

        self.cal_seq = nn.Sequential(*layers)

    def convert_printable(self, patch):
        return patch * 0 + ((patch.reshape(-1, 1) >= self.colors) * self.colors).max(dim=1)[0].reshape(patch.shape)

    def forward(self):
        return self.convert_printable(self.cal_seq(self.adv_patch.unsqueeze(dim=0))[0] + 0.5)
