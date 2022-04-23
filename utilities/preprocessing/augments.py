"""Customized augmentations, forces same augmentations to be applied to both the input and target mask.

References:
- Using functional API instead of Transforms API: https://discuss.pytorch.org/t/torchvision-transfors-how-to-perform-identical-transform-on-both-image-and-target/10606/7
- UNet-focused transforms (elastic deformation): https://github.com/hayashimasa/UNet-PyTorch/blob/main/augmentation.py
"""
import copy
import math
import random
import numpy as np
import torch
from torchvision import transforms as T
from torchvision.transforms import functional as F

# For elastic deformations.
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter

from config.model import ResNetConfig
from config.training import AugmentationConfig as acfg

class DoubleCompose(T.Compose):
    """Used to compose set of dual transforms. """
    def __call__(self, image, mask):
        for t in self.transforms:
            image, mask = t(image, mask)
        return image, mask

# Dual input operations: Applies the same transformation to both the input and target.
class PadToMinimumSize:
    """In UNet paper (https://arxiv.org/pdf/1505.04597.pdf), reflection padding is used to ensure that the input retains a valid shape that can match decoder input sizes.
    
    This function pads the input and target simultaneously based on the number of downsampling steps.

    # Since downsampling reduces dimensions by factor 2
    # Arbitrary input sizes will result in an odd dimension value
    # This results in unmatched sizes for the decoder skip connection.
    # This uses reflection padding to pad to a input size that will not become odd.
    """
    def __init__(
            self, 
            downsampling_steps=ResNetConfig.DOWNSAMPLING_STEPS,
            padding_mode="reflect",     # Whether to use resizing instead.
        ):
        self.downsampling_steps = downsampling_steps
        self.min_size = 2 ** downsampling_steps
        self.padding_mode = padding_mode
    
    def __call__(self, input, target):
        # assert input.size()[1:] == target.size()[1:], "Input and target sizes do not match."
        w, h = input.size()[1:]

        h_padded_size = int(np.ceil(h / self.min_size)) * self.min_size
        w_padded_size = int(np.ceil(w / self.min_size)) * self.min_size

        w_pad_left = int(np.floor((h_padded_size - h) / 2))
        w_pad_right = int(np.ceil((h_padded_size - h) / 2))
        h_pad_top = int(np.floor((w_padded_size - w) / 2))
        h_pad_bottom = int(np.ceil((w_padded_size - w) / 2))

        input = F.pad(
            input, 
            padding=[w_pad_left, h_pad_top, w_pad_right, h_pad_bottom], 
            padding_mode=self.padding_mode
        )
        target = F.pad(target, padding=[w_pad_left, h_pad_top, w_pad_right, h_pad_bottom], padding_mode=self.padding_mode)

        return input, target

class RandomHorizontalFilp:
    def __init__(self, p=0.5):
        self.prob = p
    
    def __call__(self, input, target):
        if random.random() < self.prob:
            input = F.hflip(input)
            target = F.hflip(target)

        return input, target

class RandomVerticalFlip:
    def __init__(self, p=0.5):
        self.prob = p
    
    def __call__(self, input, target):
        if random.random() < self.prob:
            input = F.vflip(input)
            target = F.vflip(target)

        return input, target

class RandomRotateAndCrop:
    def __init__(self, max_angle=90, p=0.5):
        self.max_angle = max_angle
        self.prob = p
    
    def __call__(self, input, target):
        if random.random() < self.prob:
            # Rotate by random angle, then crop.
            _rotation_angle = random.random() * self.max_angle
            rotator = RotateAndCrop()
            
            input = rotator(input, _rotation_angle)
            target = rotator(target, _rotation_angle)

        return input, target

class RandomCropAndResize:
    def __init__(self, maximum_scale=0.4, p=0.5):
        self.max_scale = maximum_scale
        self.prob = p
    
    def __call__(self, input, target):
        if random.random() < self.prob:
            # Keeping shape of input for resizing operation
            output_shape = input.shape

            # Retaining scale of image when cropping
            original_height_width_scale = output_shape[1] / output_shape[2]

            # Randomly select top and left
            sampled_top = random.randint(np.floor(output_shape[1] * self.max_scale), output_shape[1])
            sampled_left = random.randint(np.floor(output_shape[2] * self.max_scale), output_shape[2])

            # Randomly choose height while retaining scale
            # sampled_height = random.randint(0, sampled_top*(1-self.max_scale))
            sampled_height = int(np.round(output_shape[1] * self.max_scale))
            # To retain scale, width is simply height multipled by original scale
            sampled_width = int(np.round(sampled_height / original_height_width_scale))

            # Crop to the random size crop.
            cropped_image = F.crop(input, top=output_shape[1]-sampled_top, left=output_shape[2]-sampled_left, height=sampled_height, width=sampled_width)
            input = F.resize(cropped_image, output_shape[1:])
            # print(input.size())

            cropped_target = F.crop(target, top=output_shape[1]-sampled_top, left=output_shape[2]-sampled_left, height=sampled_height, width=sampled_width)
            # print(cropped_target.size())
            target = F.resize(cropped_target, output_shape[1:])

        return input, target

class ElasticDeform:
    """Elastic deformation as mentioned in original UNet paper.
    
    Based on implimentation from https://github.com/hayashimasa/UNet-PyTorch/blob/main/augmentation.py. Modified to fix bugs and add zoom to avoid black bars."""
    def __init__(
            self, 
            p=0.5, 
            alpha=acfg.ELASTIC_DEFORM_ALPHA_MAX, 
            sigma=acfg.ELASTIC_DEFORM_SIGMA_MAX, 
            seed=None, 
            randinit=True
        ):
        if not seed:
            seed = random.randint(1, 100)
        self.random_state = np.random.RandomState(seed)
        self.alpha = alpha
        self.sigma = sigma
        self.p = p
        self.randinit = randinit

    def __call__(self, image, mask):
        if random.random() < self.p:
            if self.randinit:
                self.random_state = np.random.RandomState(None)
                self.alpha = random.uniform(acfg.ELASTIC_DEFORM_ALPHA_MIN, acfg.ELASTIC_DEFORM_ALPHA_MAX)
                self.sigma = random.uniform(acfg.ELASTIC_DEFORM_SIGMA_MIN, acfg.ELASTIC_DEFORM_SIGMA_MAX)

            dim = image.shape
            mask_dim = mask.shape
            dx = self.alpha * gaussian_filter(
                (self.random_state.rand(*dim[1:]) * 2 - 1),
                self.sigma,
                mode="constant",
                cval=0
            )
            dy = self.alpha * gaussian_filter(
                (self.random_state.rand(*dim[1:]) * 2 - 1),
                self.sigma,
                mode="constant",
                cval=0
            )
            
            image = copy.deepcopy(image).numpy()
            mask = copy.deepcopy(mask).numpy()

            x, y = np.meshgrid(np.arange(dim[2]), np.arange(dim[1]))
            indices = np.reshape(y+dy, (-1, 1)), np.reshape(x+dx, (-1, 1))

            for ind, channel in enumerate(image):
                image[ind] = map_coordinates(channel, indices, order=1).reshape(dim[1:])
                # img_channels[ind] = channel.reshape(dim[1:])
            
            for ind, channel in enumerate(mask):
                mask[ind] = map_coordinates(channel, indices, order=1).reshape(dim[1:])

            mask = mask.reshape(mask_dim)
            image, mask = torch.from_numpy(image), torch.from_numpy(mask)

            # Add slight zoom to remove black bars on edges.
            cropped_image = F.center_crop(image, [dim[1]-5, dim[2]-5])
            image = F.resize(cropped_image, [dim[1], dim[2]])
            cropped_mask= F.center_crop(mask, [dim[1]-5, dim[2]-5])
            mask = F.resize(cropped_mask, [dim[1], dim[2]])
        return image, mask
    
# Single input operations: Can be composed into dual ops.
class RotateAndCrop:
    """Rotation with cropping of black bars.
    
    https://stackoverflow.com/questions/16702966/rotate-image-and-crop-out-black-borders"""
    def __init__(self, do_crop=True):
        self.do_crop = do_crop
        # self.interpolation_mode = interpolation_mode
    
    def __call__(self, image, rotation_angle=0):
        # output is forced to retain original shape.
        output_shape = image.size()
        # Rotate image
        image = F.rotate(image, rotation_angle)

        if self.do_crop:
            # Determine maximum width and height of crop
            crop_width, crop_height = _find_largest_rotated_rect_size(output_shape[1], output_shape[2], rotation_angle)
            # Crop centre of image by size
            cropped_image = F.center_crop(image, [int(math.floor(crop_width)), int(math.floor(crop_height))])
            image = F.resize(cropped_image, output_shape[1:])
        return image

def _find_largest_rotated_rect_size(width, height, angle):
    """Determines the maximum size of the axis-alligned rectangle after a rotation.
    
    From https://stackoverflow.com/questions/16702966/rotate-image-and-crop-out-black-borders
    """
    # Casting angle to radians
    angle = math.radians(angle)

    quadrant = int(math.floor(angle / (math.pi / 2))) & 3
    sign_alpha = angle if ((quadrant & 1) == 0) else math.pi - angle
    alpha = (sign_alpha % math.pi + math.pi) % math.pi

    bb_w = width * math.cos(alpha) + height * math.sin(alpha)
    bb_h = width * math.sin(alpha) + height * math.cos(alpha)

    gamma = math.atan2(bb_w, bb_w) if (width < height) else math.atan2(bb_w, bb_w)

    delta = math.pi - alpha - gamma

    length = height if (width < height) else width

    d = length * math.cos(alpha)
    a = d * math.sin(alpha) / math.sin(delta)

    y = a * math.cos(gamma)
    x = y * math.tan(gamma)

    return (
        bb_w - 2 * x,
        bb_h - 2 * y
    )

class AddGaussianNoise:
    def __init__(self, mu=0, sigma=1.0, p=0.5):
        self.mu = mu
        self.sigma = sigma
        self.prob = p

    def __call__(self, image):
        if random.random() < self.prob:
            # Add gaussian noise.
            image = image + torch.randn(image.size()) * self.sigma + self.mu
        return image

class PadSingleImage:
    """Same as dual transform, but designed for single image. Used for evaluations.
    
    In UNet paper (https://arxiv.org/pdf/1505.04597.pdf), reflection padding is used to ensure that the input retains a valid shape that can match decoder input sizes.
    
    This function pads the input and target simultaneously based on the number of downsampling steps.

    # Since downsampling reduces dimensions by factor 2
    # Arbitrary input sizes will result in an odd dimension value
    # This results in unmatched sizes for the decoder skip connection.
    # This uses reflection padding to pad to a input size that will not become odd.
    """
    def __init__(
            self, 
            downsampling_steps=ResNetConfig.DOWNSAMPLING_STEPS,
            padding_mode="reflect",     # Whether to use resizing instead.
        ):
        self.downsampling_steps = downsampling_steps
        self.min_size = 2 ** downsampling_steps
        self.padding_mode = padding_mode
    
    def __call__(self, input):
        w, h = input.size()[1:]

        h_padded_size = int(np.ceil(h / self.min_size)) * self.min_size
        w_padded_size = int(np.ceil(w / self.min_size)) * self.min_size

        w_pad_left = int(np.floor((h_padded_size - h) / 2))
        w_pad_right = int(np.ceil((h_padded_size - h) / 2))
        h_pad_top = int(np.floor((w_padded_size - w) / 2))
        h_pad_bottom = int(np.ceil((w_padded_size - w) / 2))

        input = F.pad(
            input, 
            padding=[w_pad_left, h_pad_top, w_pad_right, h_pad_bottom], 
            padding_mode=self.padding_mode
        )

        return input

if __name__ == "__main__":
    # This attempts to plot each transformation, showing if each works as intended
    import matplotlib.pyplot as plt
    from utilities.data import ImageTargetDataset
    from utilities.data import parse_glas_dataset

    from config.data import DataConfig as dcfg

    train, test = parse_glas_dataset(dcfg.DATA_FOLDER)

    # Test without transforms
    train_dataset = ImageTargetDataset(train)
    test_img, test_target = train_dataset[2]
    test_target = test_target.unsqueeze(0)

    # Plotting command
    def plot_comparison(input_image, input_target, transformed_image, transformed_target, title=None):
        # Cast images back to channels-last format for matplotlib
        input_image = np.transpose(input_image, [1, 2, 0])
        transformed_image = np.transpose(transformed_image, [1, 2, 0])
        input_target = np.transpose(input_target, [1, 2, 0])
        transformed_target = np.transpose(transformed_target, [1, 2, 0])

        fig, ax = plt.subplots(2, 2, figsize=(8, 8))

        ax[0, 0].imshow(input_image)
        # ax[0, 0].xaxis.set_visible(False)
        # ax[0, 0].yaxis.set_visible(False)
        ax[0, 0].title.set_text("Input Image")

        ax[0, 1].imshow(transformed_image)
        # ax[0, 1].xaxis.set_visible(False)
        # ax[0, 1].yaxis.set_visible(False)
        ax[0, 1].title.set_text("Transformed Image")

        ax[1, 0].imshow(input_target)
        # ax[1, 0].xaxis.set_visible(False)
        # ax[1, 0].yaxis.set_visible(False)
        ax[1, 0].title.set_text("Input Target")

        ax[1, 1].imshow(transformed_target)
        # ax[1, 1].xaxis.set_visible(False)
        # ax[1, 1].yaxis.set_visible(False)
        ax[1, 1].title.set_text("Transformed Target")

        if title is not None:
            fig.suptitle(title)

        plt.show()
    
    # Sanity check
    print(test_img.size())
    plot_comparison(test_img, test_target, test_img, test_target, title="Sanity Check (Images are not transformed)")

    # Test 0: Reflection Padding of Input
    augment = PadToMinimumSize(downsampling_steps=5)
    transformed_image, transformed_target = augment(test_img, test_target)
    plot_comparison(test_img, test_target, transformed_image, transformed_target, title="Reflection Padding")

    # Test 1: Random Horizontal Flipping
    augment = RandomHorizontalFilp(p=1.0)
    transformed_image, transformed_target = augment(test_img, test_target)
    plot_comparison(test_img, test_target, transformed_image, transformed_target, title="Random Horizontal Flipping")

    # Test 2: Random Vertical Flipping
    augment = RandomVerticalFlip(p=1.0)
    transformed_image, transformed_target = augment(test_img, test_target)
    plot_comparison(test_img, test_target, transformed_image, transformed_target, title="Random Vertical Flipping")

    # Test 3: Rotate and Crop
    augment = RandomRotateAndCrop(p=1.0)
    transformed_image, transformed_target = augment(test_img, test_target)
    plot_comparison(test_img, test_target, transformed_image, transformed_target, title="Rotate and Crop")

    # Test 4: Elastic Deformation
    augment = ElasticDeform(p=1.0, randinit=True)
    transformed_image, transformed_target = augment(test_img, test_target)
    
    plot_comparison(test_img, test_target, transformed_image, transformed_target, title="Elastic Deformation")

    # Test 5: Adding Gaussian Noise
    cast_to_tensor = T.ToTensor()
    augment = AddGaussianNoise(p=1.0)
    transformed_image = augment(test_img)
    transformed_target = test_target    # Gaussian noise is not added to the target mask.
    plot_comparison(test_img, test_target, transformed_image, transformed_target, title="Gaussian Noise added to Input Image")

    # Test 6: Random Crop and Resize
    augment = RandomCropAndResize(p=1.0)
    transformed_image, transformed_target = augment(test_img, test_target)
    plot_comparison(test_img, test_target, transformed_image, transformed_target, title="Random Crop and Resizing")
    
