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
                              1, 1, 1, 1, 1, 1], dims=[64, 128, 256, 512, 512, 1024])

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

    def printTensor(func: Callable) -> Callable:
        def wrapper(*args, **kwargs):
            # Get the local variables of the function
            local_vars = func.__globals__.copy()
            local_vars.update(func.__code__.co_varnames)

            # Iterate through the local variables
            for var_name, var_value in local_vars.items():
                if isinstance(var_value, torch.Tensor):
                    print(
                        f"Tensor name: '{var_name}' Shape: {var_value.shape}")

            # Call the original function and return its result
            return func(*args, **kwargs)
        return wrapper

    @printTensor
    def forward(self, x):
        x1 = x[:, :3]
        x2 = x[:, 3:]
        x = torch.cat([x1, x2], dim=0)

        outConvX = []
        for i in range(len(self.model.stages)):
            x = self.model.downsample_layers[i](x)
            x = self.model.stages[i](x)
            outConvX.append(x)
            if i == 2:
                batchSize = int(outConvX[i].shape[0]/2)
                xA = outConvX[i][:batchSize, :, :, :]
                xB = outConvX[i][batchSize:, :, :, :]
                out_conv_redir = self.conv_redir(xA)
                out_correlation = correlate(xA, xB)
                x = torch.cat([out_conv_redir, out_correlation], dim=1)
                out_conv3 = self.conv3_1(x)
                outConvX.append(out_conv3)
                x = out_conv3
        flow = []
        flowUp = []
        outDeconv = []
        concat = []
        for i in range(5):
            outConvPlus = outConvX[6-i]
            outConv = outConvX[6-(i+1)
                               ] if i <= 2 else outConvX[1][:batchSize, :, :, :]
            flow.append(self.predictFlow[i](concat[i-1]) if i >
                        0 else self.predictFlow[i](outConvPlus))
            if i == 4:
                break
            flowUp.append(crop_like(
                self.upsampledFlow[i](flow[i]), outConv))
            outDeconv.append(crop_like(self.deconv[i](
                outConvPlus) if i == 0 else self.deconv[i](concat[i-1]), outConv))
            concat.append(
                torch.cat((outConv, outDeconv[i], flowUp[i]), 1))

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
