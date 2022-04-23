import torch
from torch import nn
import torch.nn.functional as F
from torchvision import transforms as T

from config.image import ImageStatistics

from utilities.preprocessing.augments import DoubleCompose, PadToMinimumSize, ElasticDeform, RandomHorizontalFilp, RandomVerticalFlip, RandomRotateAndCrop, AddGaussianNoise, RandomCropAndResize

DEFAULT_DUAL_AUGS = DoubleCompose([
    PadToMinimumSize(),
    RandomCropAndResize(p=0.7),
    RandomHorizontalFilp(),
    RandomVerticalFlip(),
    RandomRotateAndCrop(),
    # ElasticDeform(),
])

NO_DUAL_AUGS = DoubleCompose([
    PadToMinimumSize(),
])

DEFAULT_IMAGE_AUGS = T.Compose([
    T.Normalize(ImageStatistics.CHANNEL_WISE_MEANS, ImageStatistics.CHANNEL_WISE_STDS),
    AddGaussianNoise(sigma=0.2),
    T.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5)
])

ONLY_NORMALIZE_IMAGE_AUGS = nn.Sequential(
    T.Normalize(ImageStatistics.CHANNEL_WISE_MEANS, ImageStatistics.CHANNEL_WISE_STDS),
)

CUSTOM_IMAGE_AUGS = nn.Sequential(
    T.Normalize(ImageStatistics.CHANNEL_WISE_MEANS, ImageStatistics.CHANNEL_WISE_STDS),
)

DEFAULT_MASK_AUGS = nn.Sequential(
    T.Normalize(0, 1)
)

CUSTOM_MASK_AUGS = nn.Sequential(
    T.Normalize(0, 1)
)