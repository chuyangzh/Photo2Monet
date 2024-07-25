# blocks.py

import torch.nn as nn

class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, down=True, use_act=True, **kwargs):
        super().__init__()
        
        if down:
            conv_layer = nn.Conv2d(in_channels, out_channels, padding_mode="reflect", **kwargs)
        else:
            conv_layer = nn.ConvTranspose2d(in_channels, out_channels, **kwargs)
        
        if use_act:
            act_layer = nn.ReLU(inplace=True)
        else:
            act_layer = nn.Identity()
        
        self.cnn = nn.Sequential(
            conv_layer,
            nn.InstanceNorm2d(out_channels),
            act_layer,
        )

    def forward(self, x):
        return self.cnn(x)


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            CNNBlock(channels, channels, kernel_size=3, padding=1),
            CNNBlock(channels, channels, use_act=False, kernel_size=3, padding=1),
        )

    def forward(self, x):
        return x + self.block(x)