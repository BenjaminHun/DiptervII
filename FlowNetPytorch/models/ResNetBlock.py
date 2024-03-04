import torch
import torch.nn as nn
import torchvision


class ResNetBlock(nn.Module):
    def __init__(self, inputC, outputC, cycle=1, stride=1, kernel_size=3):
        super(ResNetBlock, self).__init__()
        self.layers = []
        self.inputC = inputC
        self.outputC = outputC
        self.cycle = cycle
        self.stride = stride
        self.kernelSize = kernel_size
        self.conv2 = nn.Conv2d(in_channels=self.outputC,
                               out_channels=self.outputC, padding="same", kernel_size=self.kernelSize)
        self.conv2_2 = nn.Conv2d(in_channels=self.outputC,
                                 out_channels=self.outputC, padding="same", kernel_size=self.kernelSize)
        self.batchNorm = nn.BatchNorm2d(self.outputC)
        self.batchNorm_2 = nn.BatchNorm2d(self.outputC)
        self.reLU = nn.ReLU()
        self.dimChange = nn.Conv2d(in_channels=self.inputC,
                                   out_channels=self.outputC, stride=(self.stride, self.stride), kernel_size=self.kernelSize)
        self.shortcutSameDim = nn.Sequential()
        self.shortcutDiffDim = nn.Sequential(
            nn.Conv2d(self.inputC, self.outputC, kernel_size=1,
                      stride=stride, bias=False),
            nn.BatchNorm2d(self.outputC)
        )

    def forward(self, x):
        diffDim = self.stride != 1 or self.inputC != self.outputC
        if diffDim:
            x = residual = self.shortcutDiffDim(x)

        for i in range(self.cycle):
            if diffDim and i == 0:
                pass
            else:
                residual = self.shortcutSameDim(x)
            x = self.conv2(x)
            x = self.batchNorm(x)
            x = self.reLU(x)
            x = self.conv2_2(x)
            x = self.batchNorm_2(x)
            x += residual
            x = self.reLU(x)

        return x
