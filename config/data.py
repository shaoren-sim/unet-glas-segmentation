from utilities.augmentations import DEFAULT_DUAL_AUGS, DEFAULT_IMAGE_AUGS, CUSTOM_IMAGE_AUGS

class DataConfig:
    DATA_FOLDER = "/path/to/dataset_glas/warwick_qu_dataset_released_2016_07_08/Warwick QU Dataset (Released 2016_07_08)"

    INPUT_CHANNELS = 3
    OUTPUT_CHANNELS = 2

    # Augmentations
    DUAL_AUGMENTATIONS = DEFAULT_DUAL_AUGS         # Applied to both input images and masks (Applied on int images)
    IMAGE_SPECIFIC_AUGMENTATIONS = DEFAULT_IMAGE_AUGS     # Applied only to input images (Applied on float images)

    # Debug Options
    DATASET_PERCENTAGE = 1.0        # Only uses a subset of the available data. Each epoch has less update steps, but allows for fast debugging.