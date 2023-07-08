import torch
import torch.nn as nn
from utils.utils import normalize_tensor


class NoiseLikePatch(torch.nn.Module):
    def __init__(self, H_size, W_size):
        super().__init__()
        self.patch_H, self.patch_W = H_size, W_size
        self.adv_patch = nn.Parameter(torch.ones((3, H_size, W_size), requires_grad=True))
    
    def get_patch_size(self):
        return self.patch_H, self.patch_W
    
    def forward(self, batch_size=None):
        return self.adv_patch.clamp(0, 1)


class TpConvGenerator(nn.Module):
    def __init__(self, H_init, W_init, expand_stage=2):
        super().__init__()
        self.patch_H, self.patch_W = H_init, W_init
        self.latent_matrix = nn.Parameter(torch.rand(size=(3, H_init, W_init), requires_grad=True))

        layers = []
        for _ in range(expand_stage):
            layers.append(torch.nn.ConvTranspose2d(3, 3, kernel_size=2, stride=2))
            layers.append(nn.ReLU())

        layers.append(torch.nn.Conv2d(3, 32, kernel_size=5, padding=2))
        layers.append(nn.ReLU())
        layers.append(torch.nn.Conv2d(32, 3, kernel_size=5, padding=2))
        layers.append(nn.ReLU())
        layers.append(nn.BatchNorm2d(3))

        self.cal_seq = nn.Sequential(*layers)

    def forward(self, batch_size):
        return normalize_tensor(self.cal_seq(
            self.latent_matrix.unsqueeze(dim=0).broadcast_to((batch_size, 3, self.patch_H, self.patch_W))
        ))
