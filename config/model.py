import torch
import torch.nn as nn

class ModelConfig:
    # Supported model types: UNet, FCN (base model)
    # TODO: Maybe FCN, just disable all skip connections.
    MODEL_TYPE = "UNet"

    # Supported encoder backbones: ResNet18, ResNet34, ResNet50
    ENCODER_BACKBONE = "ResNet18"

    # Supported decoder backbones: ResNet18, ResNet34, ResNet50
    DECODER_BACKBONE = "ResNet18"

class ResNetConfig:
    """Configuration class for the ResNet backbone for the UNet model."""
    ACTIVATION_FUNCTION = nn.GELU               # Activation function for ResNet
    CHANNELS_ORDERING = [32, 64, 128, 256]     # Order of ResNet channels. Default is [64, 128, 256, 512]

    # Padding mode for encoder. circular is chosen based on https://arxiv.org/abs/2010.02178
    ENCODER_PADDING_MODE = "circular"           # Supported padding modes: ['zeros', 'reflect', 'replicate', circular']
    DECODER_PADDING_MODE = "zeros"              # Tentatively cannot be changed.

    # ResNet has 5 downsampling steps, 1 Pool, and 4 Conv2d.
    # This is used to pad the input image to ensure that encoder sizes match decoder sizes.
    DOWNSAMPLING_STEPS = 5

    # For ResNet50, bottleneck blocks expand channels by a factor of 4.
    # This can be finetuned as necessary.
    BOTTLENECK_EXPANSION = 4