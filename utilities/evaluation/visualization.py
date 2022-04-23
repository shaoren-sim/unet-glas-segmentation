"""Functions used for data evaluation."""

import numpy as np

import matplotlib.pyplot as plt
from matplotlib import colors
from PIL import Image

import torch
from torch import nn
import torchvision.transforms.functional as F

from utilities.data import instance_to_semantic

from config.data import DataConfig as dcfg
from config.model import ModelConfig as mcfg
from config.image import ImageStatistics

from utilities._checks import SUPPORTED_MODELS, SUPPORTED_ENCODER_BACKBONES, SUPPORTED_DECODER_BACKBONES
from utilities.preprocessing.augments import PadSingleImage

padding = PadSingleImage()

def prepare_model(model: nn.Module, checkpoint_path: str, device: str = "cpu"):
    """Loading checkpoint into model.

    Args:
        model (nn.module): Model architecture, should be UNet() or FCN().
        checkpoint_path (str): Path to checkpoint. Should be .pth file.
        device (str, optional): Where to load model. Defaults to "cpu".
    """    
    checkpoint = torch.load(checkpoint_path, map_location=device)["state_dict"]
    model.load_state_dict(checkpoint)
    model.to(device)

def get_padded_image_and_annotation_from_model(image, model, device):
    image = F.pil_to_tensor(image)
    padded_image = padding(image).float() / 255

    # Normalize image to match image channel statistics during training
    padded_image = F.normalize(padded_image, ImageStatistics.CHANNEL_WISE_MEANS, ImageStatistics.CHANNEL_WISE_STDS)

    with torch.no_grad():
        annotation_logits = model.segment(padded_image.to(device).unsqueeze(0))
    return padded_image.permute(1, 2, 0), np.argmax(annotation_logits.cpu(), axis=1).squeeze(0)

def return_model_predictions(image_path, model, do_crop=True, device="cpu"):
    input_image = Image.open(image_path)

    # Make model predictions
    padded_image, padded_annotations = get_padded_image_and_annotation_from_model(input_image, model, device)

    # Undoing padding via cropping.
    if do_crop:
        # Keep original dimensions to undo padding via cropping
        height, width, _ = np.array(input_image).shape
        
        unpadded_image = padded_image[
            int(np.floor((padded_image.shape[0] - height) / 2)) : int(padded_image.shape[0] - np.ceil((padded_image.shape[0] - height) / 2)),
            int(np.floor((padded_image.shape[1] - width) / 2)) : int(padded_image.shape[1] - np.ceil((padded_image.shape[1] - width) / 2)),
            :
        ]

        unpadded_annotations = padded_annotations[
            int(np.floor((padded_annotations.shape[0] - height) / 2)) : int(padded_annotations.shape[0] - np.ceil((padded_annotations.shape[0] - height) / 2)),
            int(np.floor((padded_annotations.shape[1] - width) / 2)) : int(padded_annotations.shape[1] - np.ceil((padded_annotations.shape[1] - width) / 2)),
        ]

        return unpadded_image, unpadded_annotations
    else:
        return padded_image, padded_annotations

def plot_image_and_overlay_annotation(image, annotation, alpha=0.5, show_axis=True, save_img_path=None):
    cmap = colors.ListedColormap(['lime'])

    # Ensuring that values under 1 (background) are transparent
    masked = np.ma.masked_where(annotation == 0, annotation)

    plt.imshow(image)
    plt.imshow(masked, cmap=cmap, alpha=alpha)
    if not show_axis:
        plt.axis('off')
        plt.margins(x=0)
        plt.tight_layout(pad=0)
    if save_img_path is not None:
        if not isinstance(save_img_path, str):
            raise ValueError("Output path is not a string.")
        plt.savefig(
            save_img_path, 
            bbox_inches='tight', 
            pad_inches=0,
        )
        # If save path is provided, bypass plotting.
        return
    else:
        plt.show()