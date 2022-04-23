import os
import argparse
import warnings
import pathlib

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from PIL import Image

import torch
import torchvision.transforms.functional as F

from utilities.data import instance_to_semantic

from config.data import DataConfig as dcfg
from config.model import ModelConfig as mcfg
from config.training import TrainingConfig as tcfg
from config.image import ImageStatistics

from utilities._checks import SUPPORTED_MODELS, SUPPORTED_ENCODER_BACKBONES, SUPPORTED_DECODER_BACKBONES
from utilities.preprocessing.augments import PadSingleImage
from utilities.evaluation.visualization import prepare_model, get_padded_image_and_annotation_from_model, return_model_predictions, plot_image_and_overlay_annotation

# List of image file extensions for filtering
IMAGE_FILE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".ppm", ".bmp", ".pgm", ".tif", ".tiff", ".webp")

# Loading default values from config files.
model_type = mcfg.MODEL_TYPE
encoder_backbone_type = mcfg.ENCODER_BACKBONE
decoder_backbone_type = mcfg.DECODER_BACKBONE

default_checkpoint_path = os.path.join(tcfg.CHECKPOINTING_DIRECTORY, tcfg.EXPERIMENT_NAME, "checkpoint.pth.tar")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--file", dest="path_to_evaluate", required=True, help="File/folder to run inference on.")
    parser.add_argument("-o", "--output", dest="output_path", required=False, help="Output file/folder path.")
    parser.add_argument("-c", "--checkpoint", dest="path_to_checkpoint", required=False, help=f"Path to checkpoint. Default is {default_checkpoint_path}", default=default_checkpoint_path)
    parser.add_argument("-r", "--filter", dest="filter_substring", action='append', help='Substrings to filter out of folder. Example: If -r _anno, any files with "_anno" will be ommitted during inference', required=False)
    parser.add_argument("-d", "--device", dest="device", required=False, help=f"Model inference device, can be 'cpu' or 'cuda:gpu_index'. Default is cpu.", default="cpu")
    args = parser.parse_args()

    path_to_evaluate = args.path_to_evaluate
    output_path = args.output_path
    path_to_checkpoint = args.path_to_checkpoint
    filter_substrings = args.filter_substring
    device = args.device

    # Check if inference is being run on a file or a folder.
    if os.path.isdir(path_to_evaluate):
        mode = "folder"
        # If inference mode is on a folder, output path must be provided
        if output_path is None:
            raise RuntimeError("For folder inference mode, an output path must be provided. -o /path/to/output/folder")
        
        # Creating output folder
        pathlib.Path(output_path).mkdir(parents=True, exist_ok=True) 
        print(output_path, "created.")
    elif os.path.isfile(path_to_evaluate):
        mode = "file"

        # In file inference mode, the substring filtering is ignored.
        if filter_substrings is not None:
            warnings.warn(message= "For file inference mode, the substring filtering (-r or --filter) is ignored.", category=RuntimeWarning)

        # If output path is provided as a folder, create the parent folder if does not exist
        if output_path is not None:
            parent_folder = os.path.split(output_path)[0]
            pathlib.Path(parent_folder).mkdir(parents=True, exist_ok=True) 
            print(parent_folder, "created.")
    else:
        raise RuntimeError("-f argument must be a file or a directory.")
    
    print(mode)
    print(path_to_evaluate)
    print(output_path)
    print(path_to_checkpoint)
    print(filter_substrings)
    print(device)

    # Initializing model architecture (FCN or UNet)
    model = SUPPORTED_MODELS[model_type]
    # Initializing model with selected encoder and decoder backbones
    model = model(
        encoder_backbone=SUPPORTED_ENCODER_BACKBONES[encoder_backbone_type],
        decoder_backbone=SUPPORTED_DECODER_BACKBONES[decoder_backbone_type],
    )
    print(f"Model successfully initialized.")

    # Loading checkpoint into model.
    prepare_model(model, path_to_checkpoint, device)
    print(f"Checkpoint from {path_to_checkpoint} successfully loaded.")

    # File mode - where inference is run on a single image.
    if mode == "file":
        # Load image as PIL Image for plotting
        image = Image.open(path_to_evaluate)

        _, model_annotation = return_model_predictions(path_to_evaluate, model, device=device)
        plot_image_and_overlay_annotation(image, model_annotation, show_axis=False, save_img_path=output_path)
    
    # Folder mode - where inference is run on all images in a folder.
    if mode == "folder":
        # Getting list of all image files in the folder.
        files_to_evaluate = [f for f in os.listdir(path_to_evaluate) if os.path.splitext(f)[-1] in IMAGE_FILE_EXTENSIONS]

        # Doing filtering based on the provided substring filter list
        if filter_substrings is not None:
            files_to_evaluate = [f for f in files_to_evaluate if not any(x in os.path.split(f)[-1] for x in filter_substrings)]

        for img_fn in files_to_evaluate:
            print("Processing", img_fn)
            _full_path = os.path.join(path_to_evaluate, img_fn)

            # Run model evaluation on image
            _, model_annotation = return_model_predictions(_full_path, model, device=device)
            
            # bmp is not supported by matplotlib, so casting to png
            _save_path = os.path.join(output_path, f"{os.path.splitext(img_fn)[0]}.png")
            plot_image_and_overlay_annotation(Image.open(_full_path), model_annotation, show_axis=False, save_img_path=_save_path)

            print(_save_path, "saved.")