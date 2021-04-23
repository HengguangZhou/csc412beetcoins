import torch
import torch.nn as nn
import torch.nn.functional as F
from convert import piano_roll2d_to_midi, convert_3d_to_2d, convert_2d_to_3d
import numpy as np

class res_block(nn.Module):

    def __init__(self, hidden_channels=128, kernel_size=3, pad=1):
        super(res_block, self).__init__()
        self.conv1 = nn.Conv2d(hidden_channels, hidden_channels, kernel_size=kernel_size, padding=pad)
        self.bn1 = nn.BatchNorm2d(hidden_channels)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(hidden_channels, hidden_channels, kernel_size=kernel_size, padding=pad)
        self.bn2 = nn.BatchNorm2d(hidden_channels)
        self.relu2 = nn.ReLU()

    def forward(self, x):
        h = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        return self.relu2(x + h)


class coco_decoder(nn.Module):

    def __init__(self, in_channels, out_channels=4, num_layers=16, hidden_channels=32, kernel_size=3, pad=1):
        super(coco_decoder, self).__init__()
        self.in_conv = nn.Conv2d(in_channels, hidden_channels, kernel_size=kernel_size, padding=pad)
        self.in_bn = nn.BatchNorm2d(hidden_channels)
        self.in_relu = nn.ReLU()

        self.num_layers = num_layers
        for i in range(self.num_layers - 1):
            setattr(self, f'block{i}', res_block(hidden_channels=hidden_channels, kernel_size=kernel_size, pad=pad))

        self.out_conv = nn.Conv2d(hidden_channels, out_channels, kernel_size=kernel_size, padding=pad)
        self.out_bn = nn.BatchNorm2d(out_channels)

    def forward(self, x, mask, latent, testing=False):
        x *= 1. - mask
        if testing:
            masked_original = piano_roll2d_to_midi(convert_3d_to_2d(x.detach().clone().squeeze(0).numpy(), 36))
            masked_original.save("original.mid")
        shape = x.shape
        latent = latent.reshape(shape[0], -1, shape[2], shape[3])
        x = torch.cat([x, mask, latent], dim=1)
        # x = torch.cat([x, mask], dim=1)
        x = self.in_conv(x)
        x = self.in_bn(x)
        x = self.in_relu(x)
        for i in range(self.num_layers - 1):
            x = getattr(self, f'block{i}')(x)
        x = self.out_bn(self.out_conv(x))
        return x
