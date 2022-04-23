import os
import re
import numpy as np
from PIL import Image, ImageFilter

import torch
from torchvision import transforms as T
from torchvision.transforms import functional as F
from torchvision.io import read_image
from torch.utils.data import Dataset

from typing import Tuple, List

from utilities.augmentations import DEFAULT_DUAL_AUGS, DEFAULT_IMAGE_AUGS

def natural_sort(l: List) -> List: 
    """Natural sort algorithm from https://stackoverflow.com/questions/4836710/is-there-a-built-in-function-for-string-natural-sort

    Args:
        l (List): Input list.

    Returns:
        List: Naturally sorted list.
    """    
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]

    return sorted(l, key=alphanum_key)

def parse_glas_dataset(folder_path: str) -> Tuple[List, List]:
    """Parses GLAS dataset, downloaded from https://warwick.ac.uk/fac/cross_fac/tia/data/glascontest/

    Args:
        folder_path (str): Path to the extracted folder.

    Returns:
        Tuple[List(Tuple), List(Tuple)]: List of image/segment path pairs. Split into training and test sets as defined by the image naming scheme. Full paths are returned.
    """
    all_bmps = natural_sort([os.path.join(folder_path, img) for img in os.listdir(folder_path) if os.path.splitext(img)[-1] == ".bmp"])

    # Split images and annotations from all bmp files.
    all_imgs = [img for img in all_bmps if "_anno" not in img]
    all_annotations = [img for img in all_bmps if "_anno" in img]

    # Split into train/test split.
    train_imgs = [img for img in all_imgs if "train" in img]
    test_imgs = [img for img in all_imgs if "test" in img]
    train_annotations = [img for img in all_annotations if "train" in img]
    test_annotations = [img for img in all_annotations if "test" in img]

    train_img_annot_pairs = [(img, annot) for img, annot in zip(train_imgs, train_annotations)]
    test_img_annot_pairs = [(img, annot) for img, annot in zip(test_imgs, test_annotations)]

    return train_img_annot_pairs, test_img_annot_pairs

def instance_to_semantic(segmentation_map: np.ndarray, boundary: bool = False) -> np.ndarray:
    """GLAS dataset annotations are instance segmentation maps, this function casts the instance segmentation maps to semantic segmentation maps.

    Optionally adds boundaries to the maps, allowing for semantic segmentation in-post. (Based on Apeer: https://youtu.be/VjK3hCnj8xA?t=290)

    Args:
        segmentation_map (np.ndarray): Instance segmentation map, default representation of annotations in GLAS dataset.
        boundary (bool, optional): Flag that toggles boundaries in output map. Defaults to False.

    Returns:
        np.ndarray: Semantic segmentation map.
    """

    semantic_map = (segmentation_map > 0.5).astype(int)
    if boundary:
        # Setting boundaries by finding edges
        _img = Image.fromarray(segmentation_map)
        # Finding edges
        _edges = np.array(_img.filter(ImageFilter.FIND_EDGES))
        # Cast edges to binary
        _edges = (_edges > 0.5).astype(int)

        return semantic_map + _edges
    else:
        return semantic_map

def read_bmp(bmp_path: str) -> Image:
    """Required to load images into Dataset object, as PyTorch does not support .bmp files natively."""
    return Image.open(bmp_path)

class ImageTargetDataset(Dataset):
    def __init__(
        self, 
        list_of_image_target_pair_paths, 
        dual_transforms=None,
        image_specific_transforms=None,
        target_specific_transforms=None,
        dataset_percentage=1.0
    ):
        assert 0.0 < dataset_percentage <= 1.0, "dataset_percentage must be 0 < dataset_percentage <= 1.0"
        self.image_target_pairs = list_of_image_target_pair_paths

        # if percentage is not 1.0, limit dataset to percentage.
        if dataset_percentage != 1.0:
            self.image_target_pairs = self.image_target_pairs[:int(len(self.image_target_pairs) * dataset_percentage)]

        self.dual_transforms = dual_transforms
        self.image_specific_transforms = image_specific_transforms
        self.target_specific_transforms = target_specific_transforms

    def __len__(self):
        return len(self.image_target_pairs)

    def __getitem__(self, idx):
        image_target_pair = self.image_target_pairs[idx]
        image, target = read_bmp(image_target_pair[0]), read_bmp(image_target_pair[1])
        # Convert instance segmented targets into semantic segmentation maps.
        target = instance_to_semantic(np.array(target))

        # Cast PIL images to tensor form
        image, target = F.pil_to_tensor(image), torch.from_numpy(target).unsqueeze(0)

        # Apply dual transforms, ensuring that masks are spatially accurate to transformed input images. 
        if self.dual_transforms:
            image, target = self.dual_transforms(image, target)

        # Cast image to Float Tensor, to apply image-specific transforms.
        image = image.float() / 255
        
        if self.image_specific_transforms:
            image = self.image_specific_transforms(image)
        if self.target_specific_transforms:
            target = self.target_specific_transforms(target)
        
        return image, target.squeeze(0)
    
class ChannelStatsDataset(Dataset):
    """Loads all training images without sorting, and used to compute the channel-wise mean and standard deviation for normalization purposes."""
    def __init__(self, main_dir):
        self.main_dir = main_dir
        all_imgs = [f for f in os.listdir(main_dir) if os.path.splitext(f)[-1] == ".bmp"]

        # Remove annotations from list of images
        all_imgs = [f for f in all_imgs  if "_anno" not in f]

        # Filter out training images only
        self.training_images = [f for f in all_imgs if "train" in f]

        # A transform to convert a PIL.Image into a torch Tensor
        self.transform = T.Compose([
            T.PILToTensor(),                   # Convert PIL Image to Tensor
            T.ConvertImageDtype(torch.float),  # Convert type to float
        ])

    def __len__(self):
        return len(self.training_images)

    def __getitem__(self, idx):
        img_loc = os.path.join(self.main_dir, self.training_images[idx])
        image = Image.open(img_loc)
        tensor_image = self.transform(image)
        return tensor_image

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from config.data import DataConfig as dcfg

    train, test = parse_glas_dataset(dcfg.DATA_FOLDER)

    # Test without transforms
    train_dataset = ImageTargetDataset(train)

    # Test with dual transforms
    train_dataset = ImageTargetDataset(train, DEFAULT_DUAL_AUGS, DEFAULT_IMAGE_AUGS)

    print(train_dataset[0][0].size())
    plt.imshow(train_dataset[0][0].permute(1, 2, 0))
    plt.show()