import torch
import torch.nn as nn


class ConvSkipWOBatchnorm(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1):
        super(ConvSkipWOBatchnorm, self).__init__()

        # Define layers
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size,
                              stride=stride, padding=(kernel_size-1)//2, bias=False)
        self.batchnorm = nn.BatchNorm2d(out_planes)
        self.activation = nn.LeakyReLU(0.1, inplace=True)
        self.channelWiseConv = nn.Conv2d(
            in_planes, out_planes, 1, stride) if in_planes != out_planes or stride != 1 else nn.Identity()
        print("ConvSkipWOBatchnorm")

    def forward(self, x):
        # Forward pass through convolutional layer
        out_conv = self.conv(x)

        # Forward pass through batch normalization and activation
        # out_bn = self.batchnorm(out_conv)
        out_act = self.activation(out_conv)

        # Skip connection: add input to output
        out_skip = out_act

        return out_skip+self.channelWiseConv(x)
