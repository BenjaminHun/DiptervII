import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F


class ResNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, kernel_size=3):
        super(ResNetBlock, self).__init__()

        # First convolutional layer
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=(kernel_size-1)//2, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)

        # Second convolutional layer
        self.conv2 = nn.Conv2d(out_channels, out_channels,
                               kernel_size=kernel_size, stride=1, padding=(kernel_size-1)//2, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Shortcut connection if dimensions change
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        # Main path
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        # Shortcut connection
        shortcut = self.shortcut(x)

        # Element-wise addition
        out += shortcut

        # Final ReLU activation
        out = F.relu(out)

        return out
