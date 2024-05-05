import torch
import torch.nn as nn
from torch.nn.init import kaiming_normal_, constant_

from models.ViT.VIT import ViT
from .util import conv, predict_flow, deconv, crop_like, correlate
import models

__all__ = [
    'flownetc', 'flownetc_bn'
]


class FlowNetC(nn.Module):
    expansion = 1

    def __init__(self, batchNorm=True):
        super(FlowNetC, self).__init__()
        self.config = {
            "patch_size": (2, 2),
            "hidden_size": (512, 512, 1024),
            "batch_size": 10,
            "num_hidden_layers": 4,
            "num_attention_heads": 1,
            "intermediate_size": [(4 * 512), (4*512), (4*1024)],
            "hidden_dropout_prob": 0.0,
            "attention_probs_dropout_prob": 0.0,
            "initializer_range": 0.02,
            "image_size": [(40, 56), (20, 28), (10, 14)],
            "num_classes": 10,
            "num_depths": 3,
            "num_channels": [256, 512, 512],
            "qkv_bias": True}

        self.vit = ViT(self.config)
        # self.vit = self.vit.to("cuda")

        self.batchNorm = batchNorm
        self.conv1 = conv(self.batchNorm,   3,   64, kernel_size=7, stride=2)
        self.conv2 = conv(self.batchNorm,  64,  128, kernel_size=5, stride=2)
        self.conv3 = conv(self.batchNorm, 128,  256, kernel_size=5, stride=2)
        self.conv_redir = conv(self.batchNorm, 256,   32,
                               kernel_size=1, stride=1)
        self.conv_redir_bn = nn.BatchNorm2d(32)
        self.out_correlation_bn = nn.BatchNorm2d(441)

        self.conv3_1 = conv(self.batchNorm, 473,  256)
        self.conv4 = conv(self.batchNorm, 256,  512, stride=2)
        self.conv4_1 = conv(self.batchNorm, 512,  512)
        self.conv5 = conv(self.batchNorm, 512,  512, stride=2)
        self.conv5_1 = conv(self.batchNorm, 512,  512)
        self.conv6 = conv(self.batchNorm, 512, 1024, stride=2)
        self.conv6_1 = conv(self.batchNorm, 1024, 1024)

        self.deconv5 = deconv(1024, 512)
        self.deconv4 = deconv(514, 256)
        self.deconv3 = deconv(770, 128)
        self.deconv2 = deconv(386, 64)

        self.predict_flow6 = predict_flow(1024)
        self.predict_flow5 = predict_flow(514)
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

    def forward(self, x):
        x1 = x[:, :3]
        batch_size, _, w, h = x1.shape
        x2 = x[:, 3:]

        out_conv1a = self.conv1(x1)
        # print("out_conv1a: "+str(out_conv1a.shape))
        out_conv2a = self.conv2(out_conv1a)
        # print("out_conv2a: "+str(out_conv2a.shape))
        out_conv3a = self.conv3(out_conv2a)
        # print("out_conv3a: "+str(out_conv3a.shape))

        out_conv1b = self.conv1(x2)
        # print("out_conv1b: "+str(out_conv1b.shape))
        out_conv2b = self.conv2(out_conv1b)
        # print("out_conv2b: "+str(out_conv2b.shape))
        out_conv3b = self.conv3(out_conv2b)
        # print("out_conv3b: "+str(out_conv3b.shape))

        out_conv_redir = self.conv_redir(out_conv3a)
        out_conv_redir = self.conv_redir_bn(out_conv_redir)
        # print("out_conv_redir: "+str(out_conv_redir.shape))
        out_correlation = correlate(out_conv3a, out_conv3b)
        out_correlation = self.out_correlation_bn(out_correlation)
        # print("out_correlation: "+str(out_correlation.shape))
        in_conv3_1 = torch.cat([out_conv_redir, out_correlation], dim=1)
        # print("in_conv3_1: "+str(in_conv3_1.shape))

        out_conv3 = self.conv3_1(in_conv3_1)
        # print("out_conv3: "+str(out_conv3.shape))
        vitOutputs = []
        # vitOutput = self.vit(out_conv3)
        x = out_conv3
        for i in range(self.config["num_depths"]):
            # Calculate the embedding output
            x = self.vit.embedding[i](x)
            # Calculate the encoder's output
            x = self.vit.encoder[i](x)  # 10,64,48
            x = x.transpose(-1, -2)
            newW=w//pow(2,(i+3))
            newH=h//pow(2,(i+3))
            x = x.view(
                batch_size, self.config["hidden_size"][i], newW//self.config["patch_size"][0], newH//self.config["patch_size"][1])
            vitOutputs.append(x)
        # out_conv4 = self.conv4_1(self.conv4(out_conv3))
        out_conv4 = vitOutputs[0]
        # print("out_conv4: "+str(out_conv4.shape))
        # out_conv5 = self.conv5_1(self.conv5(out_conv4))
        out_conv5 = vitOutputs[1]
        # print("out_conv5: "+str(out_conv5.shape))
        # out_conv6 = self.conv6_1(self.conv6(out_conv5))
        out_conv6 = vitOutputs[2]
        # print("out_conv6: "+str(out_conv6.shape))
# 0
        flow6 = self.predict_flow6(out_conv6)
        # print("flow6: "+str(flow6.shape))
        #flow6_up = crop_like(self.upsampled_flow6_to_5(flow6), out_conv5)
        flow6_up = self.upsampled_flow6_to_5(flow6)
        # print("flow6_up: "+str(flow6_up.shape))
        #out_deconv5 = crop_like(self.deconv5(out_conv6), out_conv5)
        out_deconv5 = self.deconv5(out_conv6)
        # print("out_deconv5: "+str(out_deconv5.shape))
        concat5 = torch.cat((out_deconv5, flow6_up), 1)
# 1
        # print("concat5: "+str(concat5.shape))
        flow5 = self.predict_flow5(concat5)
        # print("flow5: "+str(flow5.shape))
        flow5_up = crop_like(self.upsampled_flow5_to_4(flow5), out_conv4)
        # print("flow5_up: "+str(flow5_up.shape))
        out_deconv4 = crop_like(self.deconv4(concat5), out_conv4)
        # print("out_deconv4: "+str(out_deconv4.shape))
        concat4 = torch.cat((out_conv4, out_deconv4, flow5_up), 1)
# 2
        # print("concat4: "+str(concat4.shape))
        flow4 = self.predict_flow4(concat4)
        # print("flow4: "+str(flow4.shape))
        flow4_up = crop_like(self.upsampled_flow4_to_3(flow4), out_conv3)
        # print("flow4_up: "+str(flow4_up.shape))
        out_deconv3 = crop_like(self.deconv3(concat4), out_conv3)
        # print("out_deconv3: "+str(out_deconv3.shape))
        concat3 = torch.cat((out_conv3, out_deconv3, flow4_up), 1)
# 3
        # print("concat3: "+str(concat3.shape))
        flow3 = self.predict_flow3(concat3)
        # print("flow3: "+str(flow3.shape))
        flow3_up = crop_like(self.upsampled_flow3_to_2(flow3), out_conv2a)
        # print("flow3_up: "+str(flow3_up.shape))
        out_deconv2 = crop_like(self.deconv2(concat3), out_conv2a)
        # print("out_deconv2: "+str(out_deconv2.shape))
        concat2 = torch.cat((out_conv2a, out_deconv2, flow3_up), 1)
# 4
        # print("concat2: "+str(concat2.shape))
        flow2 = self.predict_flow2(concat2)
        # print("flow2: "+str(flow2.shape))

        if self.training:
            return flow2, flow3, flow4, flow5, flow6
        else:
            return flow2

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
