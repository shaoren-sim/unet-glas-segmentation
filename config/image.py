class ImageStatistics:
    # Normalization settings. Used to normalize across the training set and during evaluations.
    # To determine, run "find_mean_and_std_of_dataset.py".
    CHANNEL_WISE_MEANS = [0.7874, 0.5112, 0.7851]
    CHANNEL_WISE_STDS = [0.1594, 0.2163, 0.1188]