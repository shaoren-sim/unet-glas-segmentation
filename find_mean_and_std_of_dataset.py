"""https://stackoverflow.com/questions/60101240/finding-mean-and-standard-deviation-across-image-channels-pytorch
Script to find the channel-wise mean and standard deviation across the dataset. Will be used to normalize across the dataset in ImageStatistics.CHANNEL_MEANS and ImageStatistics.CHANNEL_STDS.
"""

import torch
from torchvision import models

from config.data import DataConfig

from utilities.data import ChannelStatsDataset

# Initializing Image DataLoader
dataset = ChannelStatsDataset(DataConfig.DATA_FOLDER)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

index_counter = 0
count_images = 0
mean = 0.0
var = 0.0

for image_batch in dataloader:
    if index_counter+1 % 100 == 0:
        print(f"Iteration {index_counter}.")
    # Rearrange batch to be the shape of [B, C, W * H]
    batch = image_batch.view(image_batch.size(0), image_batch.size(1), -1)

    # Update total number of images
    count_images += batch.size(0)

    mean += batch.mean(2).sum(0) 
    var += batch.var(2).sum(0)

    index_counter += 1

# Dividing sums by counts to obtain means and variances
mean /= count_images
var /= count_images

# Obtain standard deviation by calculating square root of variance
std = torch.sqrt(var)

print("Channel-wise means:", mean)
print("Channel-wise standard deviations:", std)

print("Copy these values into config/image.")