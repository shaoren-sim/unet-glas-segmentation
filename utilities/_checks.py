"""Lists for assertion checks."""
from models.decoders.resnet import ResNet18Decoder, ResNet34Decoder, ResNet50Decoder
from models.encoders.resnet import ResNet18, ResNet34, ResNet50_UnBottlenecked
from models.unet import UNet
from models.fcn import FCN

SUPPORTED_MODELS = {
    "UNet": UNet, 
    "FCN": FCN
}

SUPPORTED_ENCODER_BACKBONES = {
    "ResNet18": ResNet18, 
    "ResNet34": ResNet34, 
    "ResNet50": ResNet50_UnBottlenecked
}

SUPPORTED_DECODER_BACKBONES = {
    "ResNet18": ResNet18Decoder, 
    "ResNet34": ResNet34Decoder, 
    "ResNet50": ResNet50Decoder
}