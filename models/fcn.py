"""FCN approach is similar to UNet, just with the cross encoder/decoder skip connections. (see: is_unet=False argument)

FCN8s and other variants are not included. This only implements the naive variant.

Should be straight up inferior to UNet.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.decoders.resnet import ResNet18Decoder
from models.encoders.resnet import ResNet18

class FCN(nn.Module):
    def __init__(self, encoder_backbone: nn.Module = ResNet18, decoder_backbone: nn.Module = ResNet18Decoder):
        super(FCN, self).__init__()

        self.encoder = encoder_backbone(is_unet=False)
        self.decoder = decoder_backbone(is_unet=False)

        self.cross_entropy_loss = nn.CrossEntropyLoss()

    def loss_function(self, prediction, ground_truth):
        """Use Cross Entropy classification error as per original paper. Possibly can add reconstruction MSE error as well."""

        return self.cross_entropy_loss(prediction, ground_truth)

    def forward(self, x, target):
        # Encoder
        encoder_output = self.encoder(x)
        # Decoder
        decoder_output = self.decoder(*encoder_output)
        return decoder_output, target

    def segment(self, image):
        # Encoder
        encoder_output = self.encoder(image)
        # Decoder
        decoder_output = self.decoder(*encoder_output)
        return decoder_output

if __name__ == "__main__":
    # Tests to check if model can be forward-passed through correctly.
    from torchinfo import summary

    from models.encoders.resnet import ResNet18, ResNet34, ResNet50_UnBottlenecked
    from models.decoders.resnet import ResNet18Decoder, ResNet34Decoder, ResNet50Decoder

    print(summary(FCN(ResNet18, ResNet18Decoder), (8, 3, 544, 800), target=torch.randint(0, 1, (8, 544, 800))))
    print(summary(FCN(ResNet34, ResNet34Decoder), (8, 3, 544, 800), target=torch.randint(0, 1, (8, 544, 800))))
    print(summary(FCN(ResNet50_UnBottlenecked, ResNet50Decoder), (8, 3, 544, 800), target=torch.randint(0, 1, (8, 544, 800))))