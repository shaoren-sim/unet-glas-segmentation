"""Modular ResNet encoder backbone from personal WIP VQ-VAE/VAE project.

Reimplementation based on Torchvision, modified to suit modifications and link with ModelConfig.

http://pytorch.org/vision/master/_modules/torchvision/models/resnet.html#resnet18"""

import torch
from torch import nn
import torch.nn.functional as F

import numpy as np
from config.data import DataConfig

from config.model import ResNetConfig

CHANNELS_ORDERING = ResNetConfig.CHANNELS_ORDERING
# CHANNELS_ORDERING = [16, 32, 64, 128]

class ResNetBlock(nn.Module):
    expansion = 1
    
    def __init__(
            self,
            inplanes,
            planes,
            stride=1,
            downsample=None,
            base_width=CHANNELS_ORDERING[0],
            dilation=1,
            norm_layer=nn.BatchNorm2d,
        ):
        super(ResNetBlock, self).__init__()

        self.stride = stride
        self.downsample = downsample
        self.base_width = base_width
        self.norm_layer = norm_layer

        # Whether or not to use bias in conv layer, based on existence of batch norm
        if self.norm_layer is None:
            _conv_bias = True
        else:
            _conv_bias = False
        
        # Intializing convolutional layers
        self.conv1 = nn.Conv2d(
            inplanes,
            planes,
            kernel_size=3,
            stride=stride,
            bias=_conv_bias,
            padding=dilation,
            padding_mode=ResNetConfig.ENCODER_PADDING_MODE
        )
        self.conv2 = nn.Conv2d(
            planes,
            planes,
            kernel_size=3,
            stride=1,
            bias=_conv_bias,
            padding=dilation,
            padding_mode=ResNetConfig.ENCODER_PADDING_MODE
        )

        self.activation = ResNetConfig.ACTIVATION_FUNCTION()

        # Only initialize norm layers if provided.
        if self.norm_layer is not None:
            self.bn1 = norm_layer(planes)
            self.bn2 = norm_layer(planes)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        if self.norm_layer is not None:
            out = self.bn1(out)
        out = self.activation(out)
        out = self.conv2(out)
        if self.norm_layer is not None:
            out = self.bn2(out)
        # Downsample identity skip if channels do not match
        if self.downsample is not None:
            identity = self.downsample(x)
        
        # Add residual skip to output
        out = out + identity

        return out

class BottleneckBlock(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion = ResNetConfig.BOTTLENECK_EXPANSION

    def __init__(
            self,
            inplanes,
            planes,
            stride=1,
            downsample=None,
            base_width=CHANNELS_ORDERING[0],
            dilation=1,
            norm_layer=nn.BatchNorm2d,
        ):
        super(BottleneckBlock, self).__init__()

        self.stride = stride
        self.downsample = downsample
        self.base_width = base_width
        self.norm_layer = norm_layer

        # Whether or not to use bias in conv layer, based on existence of batch norm
        if self.norm_layer is None:
            _conv_bias = True
        else:
            _conv_bias = False

        width = int(planes * (base_width / CHANNELS_ORDERING[0]))
        
        # Initializing conv layers
        self.conv1 = nn.Conv2d(
            inplanes, 
            width,
            kernel_size=1,
            stride=1,
            bias=_conv_bias,
            padding_mode=ResNetConfig.ENCODER_PADDING_MODE
        )
        self.conv2 = nn.Conv2d(
            width,
            width,
            kernel_size=3,
            stride=stride,
            padding=dilation,
            dilation=dilation,
            bias=_conv_bias,
            padding_mode=ResNetConfig.ENCODER_PADDING_MODE
        )
        self.conv3 = nn.Conv2d(
            width, 
            planes * self.expansion,
            kernel_size=1,
            stride=1,
            bias=_conv_bias,
            padding_mode=ResNetConfig.ENCODER_PADDING_MODE
        )

        self.activation = ResNetConfig.ACTIVATION_FUNCTION()

        # Only initialize norm layers if provided.
        if self.norm_layer is not None:
            self.bn1 = norm_layer(width)
            self.bn2 = norm_layer(width)
            self.bn3 = norm_layer(planes * self.expansion)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        if self.norm_layer is not None:
            out = self.bn1(out)
        out = self.activation(out)
        out = self.conv2(out)
        if self.norm_layer is not None:
            out = self.bn2(out)
        out = self.activation(out)
        out = self.conv3(out)
        if self.norm_layer is not None:
            out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = out + identity
        out = self.activation(out)

        return out
    
class ResNet(nn.Module):
    def __init__(
            self,
            block,
            layers,
            zero_init_residual=False,
            width_per_group=CHANNELS_ORDERING[0],
            replace_stride_with_dilation=None,
            norm_layer=nn.BatchNorm2d,
            is_unet=False
        ):
        super(ResNet, self).__init__()

        self.norm_layer = norm_layer
        
        if self.norm_layer is None:
            self._conv_bias = True
        else:
            self._conv_bias = False

        self.inplanes = CHANNELS_ORDERING[0] // 2
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                f"or a 3-element tuple, got {replace_stride_with_dilation}"
            )
        
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(DataConfig.INPUT_CHANNELS, self.inplanes, kernel_size=7, stride=2, padding=3, bias=self._conv_bias)
        self.bn1 = norm_layer(self.inplanes)
        self.activation = ResNetConfig.ACTIVATION_FUNCTION()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, CHANNELS_ORDERING[0], layers[0])
        self.layer2 = self._make_layer(block, CHANNELS_ORDERING[1], layers[1], stride=2, dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, CHANNELS_ORDERING[2], layers[2], stride=2, dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, CHANNELS_ORDERING[3], layers[3], stride=2, dilate=replace_stride_with_dilation[2])

        # If used for UNet, output intermediate representations during forward pass
        self.is_unet = is_unet

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, BottleneckBlock):
                    nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
                elif isinstance(m, ResNetBlock):
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

    def _make_layer(
            self,
            block,
            planes,
            blocks,
            stride=1,
            dilate=False,
        ):
        norm_layer = self.norm_layer
        downsample = None

        self.dilation = 1

        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.inplanes, 
                    planes * block.expansion, 
                    kernel_size=1,
                    stride=stride,
                    bias=self._conv_bias
                ),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(
                self.inplanes, 
                planes, 
                stride=stride, 
                downsample=downsample, 
                base_width=self.base_width, 
                dilation=self.dilation,
                norm_layer=norm_layer
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    stride=1,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                )
            )

        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        # See note [TorchScript super()]
        x = self.conv1(x)
        if self.norm_layer is not None:
            x = self.bn1(x)
        x_1 = self.activation(x)
        x = self.maxpool(x_1)

        x_2 = self.layer1(x)
        x_3 = self.layer2(x_2)
        x_4 = self.layer3(x_3)
        x_5 = self.layer4(x_4)
        # print("x_1:", x_1.shape)
        # print("x_2:", x_2.shape)
        # print("x_3:", x_3.shape)
        # print("x_4:", x_4.shape)
        # print("x_5:", x_5.shape)

        return x_5, x_1, x_2, x_3, x_4

    def forward(self, x):
        x_5, x_1, x_2, x_3, x_4 = self._forward_impl(x)
        if self.is_unet:
            return x_5, x_1, x_2, x_3, x_4
        else:
            return x_5,

def ResNet18(is_unet=False):
    return ResNet(ResNetBlock, [2, 2, 2, 2], is_unet=is_unet)

def ResNet34(is_unet=False):
    return ResNet(ResNetBlock, [3, 4, 6, 3], is_unet=is_unet)

def ResNet50(is_unet=False):
    return ResNet(BottleneckBlock, [3, 4, 6, 3], is_unet=is_unet)

class ResNet50_UnBottlenecked(nn.Module):
    def __init__(
            self,
            is_unet=False
        ):
        super(ResNet50_UnBottlenecked, self).__init__()
        self.is_unet = is_unet

        self.resnet50_backbone = ResNet(BottleneckBlock, [3, 4, 6, 3], is_unet=is_unet)
        self.undo_bottleneck = nn.Conv2d(
                                    CHANNELS_ORDERING[-1]*BottleneckBlock.expansion,
                                    CHANNELS_ORDERING[-1],
                                    kernel_size=1,
                                )
    
    def forward(self, x):
        if self.is_unet:
            x_5, x_1, x_2, x_3, x_4 = self.resnet50_backbone(x)
            x_5 = self.undo_bottleneck(x_5)
            return x_5, x_1, x_2, x_3, x_4
        else:
            x_5 = self.resnet50_backbone(x)
            x_5 = self.undo_bottleneck(x_5[0])
            return x_5,

if __name__ == "__main__":
    # Tests to ensure that models can be forward-passed through properly.
    from torchinfo import summary

    # FCN mode
    print(summary(ResNet18(is_unet=False), (8, 3, 544, 800), col_names=["input_size", "output_size", "kernel_size"],))
    print(summary(ResNet34(is_unet=False), (8, 3, 544, 800), col_names=["input_size", "output_size", "kernel_size"],))
    print(summary(ResNet50_UnBottlenecked(is_unet=False), (8, 3, 544, 800), col_names=["input_size", "output_size", "kernel_size"],))

    # UNet mode
    print(summary(ResNet18(is_unet=True), (8, 3, 544, 800), col_names=["input_size", "output_size", "kernel_size"],))
    print(summary(ResNet34(is_unet=True), (8, 3, 544, 800), col_names=["input_size", "output_size", "kernel_size"],))
    print(summary(ResNet50_UnBottlenecked(is_unet=True), (8, 3, 544, 800), col_names=["input_size", "output_size", "kernel_size"],))