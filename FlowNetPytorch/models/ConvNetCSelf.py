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

        self.model = ConvNeXt(in_chans=3, depths=[
                              1, 1, 3], dims=[64, 128, 256])
        self.model2 = ConvNeXt(in_chans=256, depths=[1, 1, 1], dims=[
            512, 512, 1024])

        self.batchNorm = batchNorm
        self.conv_redir = conv(self.batchNorm, 256,   32,
                               kernel_size=1, stride=1)
        self.conv3_1 = conv(self.batchNorm, 473,  256)

        self.deconvDims = [(1024, 512), (1026, 256), (770, 128), (386, 64)]
        self.deconv = []
        for i in range(len(self.deconvDims)):
            self.deconv.append(
                deconv(self.deconvDims[i][0], self.deconvDims[i][1]).to("cuda"))

        self.predictFlowDims = [1024, 1026, 770, 386, 194]
        self.predictFlow = []
        for i in range(len(self.predictFlowDims)):
            self.predictFlow.append(predict_flow(
                self.predictFlowDims[i]).to("cuda"))

        self.upsampledFlow = []
        for _ in range(4):
            self.upsampledFlow.append(nn.ConvTranspose2d(
                2, 2, 4, 2, 1, bias=False).to("cuda"))

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                kaiming_normal_(m.weight, 0.1)
                if m.bias is not None:
                    constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                constant_(m.weight, 1)
                constant_(m.bias, 0)

    # @#printTensor
    def forward(self, x):
        x1 = x[:, :3].to('cuda')

        # print("x1 "+str(x1.shape))
        x2 = x[:, 3:].to('cuda')
        # print("x2 "+str(x2.shape))
        # x = torch.cat([x1, x2], dim=0)
        # print("x "+str(x.shape))

        outConvX = []
        outConvXa = [None]*3
        outConvXb = [None]*3
        for i in range(len(self.model.stages)):
            x1 = self.model.downsample_layers[i](x1)
            x1 = self.model.stages[i](x1)
            outConvXa[i] = x1
        for i in range(len(self.model.stages)):
            x2 = self.model.downsample_layers[i](x2)
            x2 = self.model.stages[i](x2)
            outConvXb[i] = x2
        for i in range(3):
            outConvX.append([outConvXa[i], outConvXb[i]])
        out_conv_redir = self.conv_redir(outConvXa[2])
        out_correlation = correlate(outConvXa[2], outConvXb[2])

        in_conv3_1 = torch.cat([out_conv_redir, out_correlation], dim=1)
        out_conv3 = self.conv3_1(in_conv3_1)
        x = out_conv3
        outConvX.append(x)
        for i in range(len(self.model2.stages)):
            x = self.model2.downsample_layers[i](x)
            x = self.model2.stages[i](x)
            outConvX.append(x)

        flow = []
        flowUp = []
        outDeconv = []
        concat = []

        i = 0
        outConvPlus = outConvX[6-i]
        outConv = outConvX[6-(i+1)]
        flow.append(self.predictFlow[i](outConvPlus))
        flowUp.append(crop_like(self.upsampledFlow[i](flow[i]), outConv))
        outDeconv.append(crop_like(self.deconv[i](outConvPlus), outConv))
        concat.append(torch.cat((outConv, outDeconv[i], flowUp[i]), 1))

        i = 1
        outConvPlus = outConvX[6-i]
        outConv = outConvX[6-(i+1)]
        flow.append(self.predictFlow[i](concat[i-1]))
        flowUp.append(crop_like(self.upsampledFlow[i](flow[i]), outConv))
        outDeconv.append(crop_like(self.deconv[i](concat[i-1]), outConv))
        concat.append(torch.cat((outConv, outDeconv[i], flowUp[i]), 1))

        i = 2
        outConvPlus = outConvX[6-i]
        outConv = outConvX[6-(i+1)]
        flow.append(self.predictFlow[i](concat[i-1]))
        flowUp.append(crop_like(self.upsampledFlow[i](flow[i]), outConv))
        outDeconv.append(crop_like(self.deconv[i](concat[i-1]), outConv))
        concat.append(torch.cat((outConv, outDeconv[i], flowUp[i]), 1))

        i = 3
        outConvPlus = outConvX[6-i]
        outConv = outConvX[1][0]
        flow.append(self.predictFlow[i](concat[i-1]))
        flowUp.append(crop_like(self.upsampledFlow[i](flow[i]), outConv))
        outDeconv.append(crop_like(self.deconv[i](concat[i-1]), outConv))
        concat.append(torch.cat((outConv, outDeconv[i], flowUp[i]), 1))

        i = 4
        outConvPlus = outConvX[6-i]
        outConv = outConvX[1][0]
        flow.append(self.predictFlow[i](concat[i-1]))

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
