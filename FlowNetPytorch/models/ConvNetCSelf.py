import torch
import torch.nn as nn
from torch.nn.init import kaiming_normal_, constant_
from .util import conv, predict_flow, deconv, crop_like, correlate
from .ConvNext import ConvNeXt
from typing import Any, Callable
__all__ = [
    'flownetc', 'flownetc_bn'
]


class FlowNetC(nn.Module):
    expansion = 1

    def __init__(self, batchNorm=True):
        super(FlowNetC, self).__init__()
        #self.model = ConvNeXt(in_chans=3, depths=[1, 1, 1, 1, 1, 1], dims=[
         #                     64, 128, 256, 512, 512, 1024])
        
        self.model = ConvNeXt(in_chans=3, depths=[1, 1, 1], dims=[
                              64, 128, 256])

        self.upperEncoder = nn.ModuleList()

        self.conv4 = conv(self.batchNorm, 256,  512, stride=2)
        self.conv4_1 = conv(self.batchNorm, 512,  512)
        self.upperEncoder.append(nn.Sequential(self.conv4, self.conv4_1))
        self.conv5 = conv(self.batchNorm, 512,  512, stride=2)
        self.conv5_1 = conv(self.batchNorm, 512,  512)
        self.upperEncoder.append(nn.Sequential(self.conv5, self.conv5_1))
        self.conv6 = conv(self.batchNorm, 512, 1024, stride=2)
        self.conv6_1 = conv(self.batchNorm, 1024, 1024)
        self.upperEncoder.append(nn.Sequential(self.conv6, self.conv6_1))

        self.batchNorm = batchNorm
        self.conv_redir = conv(self.batchNorm, 256,   32,
                               kernel_size=1, stride=1)
        self.conv3_1 = conv(self.batchNorm, 473,  256)

        self.deconvDims = [(1024, 512), (1026, 256), (770, 128), (386, 64)]
        self.deconv = nn.ModuleList()
        for i in range(len(self.deconvDims)):
            self.deconv.append(
                deconv(self.deconvDims[i][0], self.deconvDims[i][1]))

        self.predictFlowDims = [1024, 1026, 770, 386, 194]
        self.predictFlow = nn.ModuleList()
        for i in range(len(self.predictFlowDims)):
            self.predictFlow.append(predict_flow(
                self.predictFlowDims[i]))

        self.upsampledFlow = nn.ModuleList()
        for _ in range(4):
            self.upsampledFlow.append(nn.ConvTranspose2d(
                2, 2, 4, 2, 1, bias=False))

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                print(m)
                kaiming_normal_(m.weight, 0.1)
                if m.bias is not None:
                    constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                constant_(m.weight, 1)
                constant_(m.bias, 0)

    def forward(self, x):
        x1 = x[:, :3]
        x2 = x[:, 3:]
        batchSize = x1.shape[0]
        x = torch.cat([x1, x2], dim=0)
        saveableOutConv = [1, 3, 4, 5, 6]

        outConvX = []
        for i in range(len(self.model.stages)):  # vagy ez vagy az
            if i<3:
                x = self.model.downsample_layers[i](x)
                x = self.model.stages[i](x)
                if i == 2:
                    # indexeket addig kell felsorolni amig nem akarom az egészet
                    xA = x[:batchSize, :]
                    xB = x[batchSize:, :]
                    out_conv_redir = self.conv_redir(xA)
                    out_correlation = correlate(xA, xB)
                    x = torch.cat([out_conv_redir, out_correlation], dim=1)
                    out_conv3 = self.conv3_1(x)
                    outConvX.append(out_conv3)
                    x = out_conv3
            else:
                x=self.upperEncoder[i-3](x)

            if i in saveableOutConv:
                outConvX.append(x[:batchSize, :]if i == 1 else x)

        flow = []
        flowUp = []
        outDeconv = []
        concat = []
        for i in range(5):
            outConvPlus = outConvX[4-i]
            flow.append(self.predictFlow[i](concat[i-1]) if i >
                        0 else self.predictFlow[i](outConvPlus))
            if i == 4:
                break
            outConv = outConvX[4-(i+1)] if i < 3 else outConvX[0]
            flowUp.append(crop_like(
                self.upsampledFlow[i](flow[i]), outConv))
            outDeconv.append(crop_like(self.deconv[i](
                outConvPlus) if i == 0 else self.deconv[i](concat[i-1]), outConv))
            concat.append(
                torch.cat((outConv, outDeconv[i], flowUp[i]), 1))
            # conv batchnorm relu->conv relu batchnorm
            # conv redir->nem kell relu, sima conv meg layernorm
            # conv3_1->1 méretű convnext blokkot, majd 1x1-es conv segitségével lehuzni a méretet
            # convnext-nél stem kernel 2 legyen mert a stride is 2
            # jegyzet 2.7,  +2 link transposedconv helyett ez+conv 3x3 1x1-es padding

        if self.training:
            return flow[4], flow[3], flow[2], flow[1], flow[0]
        else:
            return flow[4]

    def weight_parameters(self):
        return [param for name, param in self.named_parameters() if 'weight' in name]

    def bias_parameters(self):
        return [param for name, param in self.named_parameters() if 'bias' in name]


def flownetc(data=None):
    """FlowNetS model architecture from the
    "Learning Optical Flow with Convolutional Networks" paper (https://arxiv.org/abs/1504.06852)

    Args:
        data : pretrained weights of the network. will create a new one if not set
    """
    model = FlowNetC(batchNorm=False)
    if data is not None:
        model.load_state_dict(data['state_dict'])
    return model


def flownetc_bn(data=None):
    """FlowNetS model architecture from the
    "Learning Optical Flow with Convolutional Networks" paper (https://arxiv.org/abs/1504.06852)

    Args:
        data : pretrained weights of the network. will create a new one if not set
    """
    model = FlowNetC(batchNorm=True)
    if data is not None:
        model.load_state_dict(data['state_dict'])
    return model
