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
            print(id(self.deconv[i]))

        self.predictFlowDims = [1024, 1026, 770, 386, 194]
        self.predictFlow = []
        for i in range(len(self.predictFlowDims)):
            self.predictFlow.append(predict_flow(
                self.predictFlowDims[i]).to("cuda"))
            print(id(self.predictFlow[i]))

        self.upsampledFlow = []
        for i in range(4):
            self.upsampledFlow.append(nn.ConvTranspose2d(
                2, 2, 4, 2, 1, bias=False).to("cuda"))
            print(id(self.upsampledFlow[i]))

        

        self.deconv5 = deconv(1024, 512)
        self.deconv4 = deconv(1026, 256)
        self.deconv3 = deconv(770, 128)
        self.deconv2 = deconv(386, 64)

        self.predict_flow6 = predict_flow(1024)
        self.predict_flow5 = predict_flow(1026)
        self.predict_flow4 = predict_flow(770)
        self.predict_flow3 = predict_flow(386)
        self.predict_flow2 = predict_flow(194)

        self.upsampled_flow6_to_5 = nn.ConvTranspose2d(
            2, 2, 4, 2, 1, bias=False)
        self.upsampled_flow5_to_4 = nn.ConvTranspose2d(
            2, 2, 4, 2, 1, bias=False)
        self.upsampled_flow4_to_3 = nn.ConvTranspose2d(
            2, 2, 4, 2, 1, bias=False)
        self.upsampled_flow3_to_2 = nn.ConvTranspose2d(
            2, 2, 4, 2, 1, bias=False)

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
        flow6 = self.predict_flow6(outConvPlus)
        flow6Up = crop_like(self.upsampled_flow6_to_5(flow6), outConv)
        outDeconv5 = crop_like(self.deconv5(outConvPlus), outConv)
        concat5 = torch.cat((outConv, outDeconv5, flow6Up), 1)

        i = 1
        outConvPlus = outConvX[6-i]
        outConv = outConvX[6-(i+1)]
        flow5 = self.predict_flow5(concat5)
        flow5Up = crop_like(self.upsampled_flow5_to_4(flow5), outConv)
        outDeconv4 = crop_like(self.deconv4(concat5), outConv)
        concat4 = torch.cat((outConv, outDeconv4, flow5Up), 1)

        i = 2
        outConvPlus = outConvX[6-i]
        outConv = outConvX[6-(i+1)]
        flow4 = self.predict_flow4(concat4)
        flow4Up = crop_like(self.upsampled_flow4_to_3(flow4), outConv)
        outDeconv3 = crop_like(self.deconv3(concat4), outConv)
        concat3 = torch.cat((outConv, outDeconv3, flow4Up), 1)

        i = 3
        outConvPlus = outConvX[6-i]
        outConv = outConvX[1][0]
        flow3 = self.predict_flow3(concat3)
        flow3Up = crop_like(self.upsampled_flow3_to_2(flow3), outConv)
        outDeconv2 = crop_like(self.deconv2(concat3), outConv)
        concat2 = torch.cat((outConv, outDeconv2, flow3Up), 1)

        i = 4
        outConvPlus = outConvX[6-i]
        outConv = outConvX[1][0]
        flow2 = self.predict_flow2(concat2)

        if self.training:
            return flow2, flow3, flow4, flow5, flow6
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
