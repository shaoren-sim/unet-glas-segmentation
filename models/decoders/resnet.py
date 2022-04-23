"""More modular ResNet encoder backbone from personal WIP VQ-VAE/VAE project. Modified to include concatenatable multi-stage filters as per-UNet.

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
CHANNELS_ORDERING = CHANNELS_ORDERING[::-1]

class ResNetDecBlock(nn.Module):
    expansion = 1
    
    def __init__(
            self,
            inplanes,
            planes,
            stride=1,
            upsample=None,
            base_width=64,
            dilation=1,
            norm_layer=nn.BatchNorm2d,
        ):
        super(ResNetDecBlock, self).__init__()

        self.stride = stride
        self.upsample = upsample
        self.base_width = base_width
        self.norm_layer = norm_layer

        # Whether or not to use bias in conv layer, based on existence of batch norm
        if self.norm_layer is None:
            _conv_bias = True
        else:
            _conv_bias = False

        if stride == 1:
            _output_dilation = 0
        else:
            _output_dilation = 1
        
        # Intializing convolutional layers
        self.conv1 = nn.ConvTranspose2d(
            inplanes,
            planes,
            kernel_size=3,
            stride=stride,
            bias=_conv_bias,
            padding=dilation,
            padding_mode=ResNetConfig.DECODER_PADDING_MODE,
            output_padding=_output_dilation,
        )
        self.conv2 = nn.Conv2d(
            planes,
            planes,
            kernel_size=1,
            stride=1,
            bias=_conv_bias,
            padding=0,
            padding_mode=ResNetConfig.DECODER_PADDING_MODE,
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
        # upsample identity skip if channels do not match
        if self.upsample is not None:
            identity = self.upsample(identity)
        
        # Add residual skip to output
        out = out + identity

        return out

class InvertedBottleneckBlock(nn.Module):
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
            upsample=None,
            base_width=ResNetConfig.CHANNELS_ORDERING[0],
            dilation=1,
            norm_layer=nn.BatchNorm2d,
        ):
        super(InvertedBottleneckBlock, self).__init__()

        self.stride = stride
        self.upsample = upsample
        self.base_width = base_width
        self.norm_layer = norm_layer

        # Whether or not to use bias in conv layer, based on existence of batch norm
        if self.norm_layer is None:
            _conv_bias = True
        else:
            _conv_bias = False

        if stride == 1:
            _output_dilation = 0
        else:
            _output_dilation = 1

        width = int(planes * (base_width / ResNetConfig.CHANNELS_ORDERING[0]))
        
        # Initializing conv layers
        self.conv1 = nn.Conv2d(
            inplanes, 
            width,
            kernel_size=1,
            stride=1,
            bias=_conv_bias,
            padding_mode=ResNetConfig.DECODER_PADDING_MODE,
        )
        self.conv2 = nn.ConvTranspose2d(
            width,
            width,
            kernel_size=3,
            stride=stride,
            padding=dilation,
            dilation=dilation,
            bias=_conv_bias,
            padding_mode=ResNetConfig.DECODER_PADDING_MODE,
            output_padding=_output_dilation,
        )
        self.conv3 = nn.Conv2d(
            width, 
            planes * self.expansion,
            kernel_size=1,
            stride=1,
            bias=_conv_bias,
            padding_mode=ResNetConfig.DECODER_PADDING_MODE,
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

        if self.upsample is not None:
            identity = self.upsample(identity)

        out = out + identity
        out = self.activation(out)

        return out
    
class ResNetDecoder(nn.Module):
    def __init__(
            self,
            block,
            layers,
            zero_init_residual=False,
            width_per_group=64,
            replace_stride_with_dilation=None,
            norm_layer=nn.BatchNorm2d,
            is_unet=False
        ):
        super(ResNetDecoder, self).__init__()
        
        self.block = block
        self.norm_layer = norm_layer
        
        if self.norm_layer is None:
            self._conv_bias = True
        else:
            self._conv_bias = False

        self.channels_before_bottleneck = ResNetConfig.CHANNELS_ORDERING[-1]

        self.inplanes = self.channels_before_bottleneck
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

        self.bn1 = norm_layer(self.inplanes)
        self.activation = ResNetConfig.ACTIVATION_FUNCTION()
        self.layer1 = self._make_layer(block, CHANNELS_ORDERING[0], layers[0], stride=1)
        self.layer2 = self._make_layer(block, CHANNELS_ORDERING[1], layers[1], stride=2, dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, CHANNELS_ORDERING[2], layers[2], stride=2, dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, CHANNELS_ORDERING[3], layers[3], stride=2, dilate=replace_stride_with_dilation[2])

        # If used for UNet, add conv layers to merge down concatenation from encoder skip connections.
        self.is_unet = is_unet
        if self.is_unet:
            self.x4_downsample = nn.Conv2d(
                CHANNELS_ORDERING[1]*2*block.expansion,
                CHANNELS_ORDERING[1]*block.expansion,
                kernel_size=1
            )
            self.x3_downsample = nn.Conv2d(
                CHANNELS_ORDERING[2]*2*block.expansion,
                CHANNELS_ORDERING[2]*block.expansion,
                kernel_size=1
            )
            self.x2_downsample = nn.Conv2d(
                CHANNELS_ORDERING[3]*2*block.expansion,
                CHANNELS_ORDERING[3]*block.expansion,
                kernel_size=1
            )
            self.x1_downsample = nn.Conv2d(
                CHANNELS_ORDERING[3],
                CHANNELS_ORDERING[3] // 2,
                kernel_size=1
            )
        self.final_conv = nn.ConvTranspose2d(
            CHANNELS_ORDERING[-1] * block.expansion,
            CHANNELS_ORDERING[-1] // 2,
            kernel_size=3,
            stride=2,
            padding=1,
            padding_mode=ResNetConfig.DECODER_PADDING_MODE,
            output_padding=1,
            bias=self._conv_bias
        )
        self.bn_final = norm_layer(CHANNELS_ORDERING[-1] // 2)
        
        self.cast_to_image = nn.ConvTranspose2d(
            CHANNELS_ORDERING[-1] // 2,
            DataConfig.OUTPUT_CHANNELS,
            kernel_size=3,
            stride=2,
            padding=1,
            padding_mode=ResNetConfig.DECODER_PADDING_MODE,
            output_padding=1,
            bias=self._conv_bias
        )
        # self.sigmoid_constraint = nn.Sigmoid()

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
                if isinstance(m, InvertedBottleneckBlock):
                    nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
                elif isinstance(m, ResNetDecBlock):
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
        upsample = None

        self.dilation = 1

        if dilate:
            self.dilation *= stride
            stride = 1

        if stride == 1:
            _output_padding = 0
        else:
            _output_padding = 1
        
        if stride != 1 or self.inplanes != planes * block.expansion:
            upsample = nn.Sequential(
                nn.ConvTranspose2d(
                    self.inplanes, 
                    planes * block.expansion, 
                    kernel_size=1,
                    stride=stride,
                    bias=self._conv_bias,
                    output_padding=_output_padding,
                ),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(
                self.inplanes, 
                planes, 
                stride=stride, 
                upsample=upsample, 
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

    def _forward_impl(self, x_5, x_1, x_2, x_3, x_4):
        if self.norm_layer is not None:
            x_5 = self.bn1(x_5)
        x = self.activation(x_5)
        x = self.layer1(x)
        x = self.layer2(x)
        if self.is_unet:
            x = torch.cat([x, x_4], 1)
            x = self.x4_downsample(x)
        x = self.layer3(x)
        if self.is_unet:
            x = torch.cat([x, x_3], 1)
            x = self.x3_downsample(x)
        x = self.layer4(x)
        if self.is_unet:
            x = torch.cat([x, x_2], 1)
            x = self.x2_downsample(x)
        x = self.final_conv(x)
        if self.is_unet:
            x = torch.cat([x, x_1], 1)
            x = self.x1_downsample(x)
        x = self.bn_final(x)
        x = self.activation(x)

        # Recast output to image
        x = self.cast_to_image(x)
        # x = self.sigmoid_constraint(x)

        return x

    def forward(self, x, x_1=0, x_2=0, x_3=0, x_4=0):
        return self._forward_impl(x, x_1, x_2, x_3, x_4)

# class ResNet18Decoder(nn.Module):
#     def __init__(self):
#         super(ResNet18Decoder, self).__init__()

#         # Popping output FC layer
#         self.model = ResNetDecoder(ResNetDecBlock, [2, 2, 2, 2])
    
#     def forward(self, x):
#         return self.model(x)

def ResNet18Decoder(is_unet=False):
    return ResNetDecoder(ResNetDecBlock, [2, 2, 2, 2], is_unet=is_unet)

def ResNet34Decoder(is_unet=False):
    return ResNetDecoder(ResNetDecBlock, [3, 4, 6, 3], is_unet=is_unet)

def ResNet50Decoder(is_unet=False):
    return ResNetDecoder(InvertedBottleneckBlock, [3, 4, 6, 3], is_unet=is_unet)

if __name__ == "__main__":
    from torchinfo import summary

    x_1=torch.randn(8, ResNetConfig.CHANNELS_ORDERING[0]//2, 272, 400)
    x_2=torch.randn(8, ResNetConfig.CHANNELS_ORDERING[0], 136, 200)
    x_3=torch.randn(8, ResNetConfig.CHANNELS_ORDERING[1], 68, 100)
    x_4=torch.randn(8, ResNetConfig.CHANNELS_ORDERING[2], 34, 50)

    for decoder in [ResNet18Decoder, ResNet34Decoder]:
        # FCN Mode
        print("FCN Mode,", decoder)
        model = decoder(is_unet=False)
        print(
            summary(
                model, 
                (8, ResNetConfig.CHANNELS_ORDERING[-1], 17, 25),
                col_names=["input_size", "output_size", "kernel_size"],
                x_1=x_1,
                x_2=x_2,
                x_3=x_3,
                x_4=x_4,
            )
        )

        # UNet Mode
        print("UNet Mode,", decoder)
        model = decoder(is_unet=True)
        print(
            summary(
                model, 
                (8, ResNetConfig.CHANNELS_ORDERING[-1], 17, 25),
                col_names=["input_size", "output_size", "kernel_size"],
                x_1=x_1,
                x_2=x_2,
                x_3=x_3,
                x_4=x_4,
            )
        )
    
    # For ResNet50 mode, due to bottleneck layers, different inputs are required.
    x_1=torch.randn(8, ResNetConfig.CHANNELS_ORDERING[0]//2, 272, 400)
    x_2=torch.randn(8, ResNetConfig.CHANNELS_ORDERING[0]*ResNetConfig.BOTTLENECK_EXPANSION, 136, 200)
    x_3=torch.randn(8, ResNetConfig.CHANNELS_ORDERING[1]*ResNetConfig.BOTTLENECK_EXPANSION, 68, 100)
    x_4=torch.randn(8, ResNetConfig.CHANNELS_ORDERING[2]*ResNetConfig.BOTTLENECK_EXPANSION, 34, 50)

    print("FCN Mode,", ResNet50Decoder)
    model = ResNet50Decoder(is_unet=False)
    print(
        summary(
            model, 
            (8, ResNetConfig.CHANNELS_ORDERING[-1], 17, 25),
            col_names=["input_size", "output_size", "kernel_size"],
            x_1=x_1,
            x_2=x_2,
            x_3=x_3,
            x_4=x_4,
        )
    )

    # UNet Mode
    print("UNet Mode,", ResNet50Decoder)
    model = ResNet50Decoder(is_unet=True)
    print(
        summary(
            model, 
            (8, ResNetConfig.CHANNELS_ORDERING[-1], 17, 25),
            col_names=["input_size", "output_size", "kernel_size"],
            x_1=x_1,
            x_2=x_2,
            x_3=x_3,
            x_4=x_4,
        )
    )