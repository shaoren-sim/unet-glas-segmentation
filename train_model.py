import torch
from torch import nn
from torch.nn import functional as F

import os
import shutil
import time
import datetime

from config.training import TrainingConfig as tcfg
from config.data import DataConfig as dcfg
from config.model import ModelConfig as mcfg

from utilities._checks import SUPPORTED_MODELS, SUPPORTED_ENCODER_BACKBONES, SUPPORTED_DECODER_BACKBONES
from utilities.augmentations import NO_DUAL_AUGS, ONLY_NORMALIZE_IMAGE_AUGS

from utilities.data import ImageTargetDataset, parse_glas_dataset
from utilities.training.training_utils import Checkpointing, csv_logger

from models.unet import UNet
from models.fcn import FCN

# Loading default values from config files.
model_type = mcfg.MODEL_TYPE
encoder_backbone_type = mcfg.ENCODER_BACKBONE
decoder_backbone_type = mcfg.DECODER_BACKBONE
experiment_name = tcfg.EXPERIMENT_NAME

device = tcfg.DEVICE
checkpoint_filepath = os.path.join(tcfg.CHECKPOINTING_DIRECTORY, tcfg.EXPERIMENT_NAME)

# if checkpoint directory is not created
if not os.path.isdir(tcfg.CHECKPOINTING_DIRECTORY):
    os.mkdir(tcfg.CHECKPOINTING_DIRECTORY)

csv_log_path_perepoch = os.path.join(checkpoint_filepath, 'logs.csv')
csv_log_path_periterstep = os.path.join(checkpoint_filepath, 'logs_per_iter_steps.csv')

assert model_type in SUPPORTED_MODELS, f"ModelConfig.MODEL_TYPE must be in {list(SUPPORTED_MODELS.keys())}"
assert encoder_backbone_type in SUPPORTED_ENCODER_BACKBONES, f"ModelConfig.SUPPORTED_ENCODER_BACKBONES must be in {list(SUPPORTED_ENCODER_BACKBONES.keys())}"
assert decoder_backbone_type in SUPPORTED_DECODER_BACKBONES, f"ModelConfig.SUPPORTED_DECODER_BACKBONES must be in {list(SUPPORTED_DECODER_BACKBONES.keys())}"

# Initializing model architecture (FCN or UNet)
model = SUPPORTED_MODELS[model_type]
# Initializing model with selected encoder and decoder backbones
model = model(
    encoder_backbone=SUPPORTED_ENCODER_BACKBONES[encoder_backbone_type],
    decoder_backbone=SUPPORTED_DECODER_BACKBONES[decoder_backbone_type],
)

# Logic for resuming from prior training.
_resume = False
if os.path.isdir(checkpoint_filepath):
    # If folder already exists, but 
    if not tcfg.RESUME_LATEST:
        raise ValueError(f"{checkpoint_filepath} already exists. If want to resume from prior training, set RESUME_LATEST = True.")
    else:
        _resume = True
        print("Resuming from latest checkpoint.")
        ckpt_path = os.path.join(checkpoint_filepath, "checkpoint.pth.tar")
        shutil.copyfile(ckpt_path, os.path.join(checkpoint_filepath, "checkpoint_bkp.pth.tar"))
        ckpt_dict = torch.load(ckpt_path, map_location=device)
        _start_epoch = ckpt_dict["epoch"] + 1
        model.load_state_dict(ckpt_dict["state_dict"])
        _optimizer_state_dict = ckpt_dict["optimizer"]
        # checkpoint_filepath = f"{checkpoint_filepath}_{datetime.datetime.today().strftime('%Y%m%d-%H%M%S')}"
else:
    _start_epoch = 0

# Initialize batch size based on whether or not gradient accumulation needs to be done.
if tcfg.BATCH_SIZE > 1:
    if not tcfg.DO_GRADIENT_ACCUMULATION:
        print("batch size is > 1, to use higher batch sizes, set TrainingConfig.DO_GRADIENT_ACCUMULATION to True. This issue occurs because of the inconsistent batch sizes in the dataset.")
    else:
        print("Gradient accumulation active.")

raw_batch_size = tcfg.BATCH_SIZE
if tcfg.DO_GRADIENT_ACCUMULATION:
    batch_size = raw_batch_size
elif not tcfg.DO_GRADIENT_ACCUMULATION:
    batch_size = 1

print("Batch size per update step:", batch_size)

# Initializing dataset
train_set, test_set = parse_glas_dataset(dcfg.DATA_FOLDER)
if tcfg.DO_AUGMENTATIONS:
    train_dataset = ImageTargetDataset(
        list_of_image_target_pair_paths=train_set, 
        dataset_percentage=dcfg.DATASET_PERCENTAGE, 
        dual_transforms=dcfg.DUAL_AUGMENTATIONS, 
        image_specific_transforms=dcfg.IMAGE_SPECIFIC_AUGMENTATIONS
    )
else:
    train_dataset = ImageTargetDataset(
        list_of_image_target_pair_paths=train_set, 
        dataset_percentage=dcfg.DATASET_PERCENTAGE,
        dual_transforms=NO_DUAL_AUGS,        # NO_DUAL_AUGS includes the input padding operation.
        image_specific_transforms=ONLY_NORMALIZE_IMAGE_AUGS,
    )

train_dataset_size = len(train_dataset)
print(train_dataset_size, "images in training dataset.")

train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=True, pin_memory=True, num_workers=1)

if tcfg.DO_VALIDATION:
    valid_dataset = ImageTargetDataset(
        list_of_image_target_pair_paths=test_set,
        dataset_percentage=dcfg.DATASET_PERCENTAGE,
        dual_transforms=NO_DUAL_AUGS        # Validation set does not use augmentations.
    )
    valid_dataset_size = len(valid_dataset)
    print(valid_dataset_size, "images in validation dataset.")

    valid_dataloader = torch.utils.data.DataLoader(valid_dataset, batch_size=1, shuffle=True, pin_memory=True, num_workers=1)

# Sanity check.
print("Running dry update pass to test training loop.")
with torch.no_grad():
    test_images, test_targets = next(iter(train_dataloader))
    print("Test input batch:", test_images.size())
    print("Test target batch:", test_targets.size())

    test_output = model(test_images, test_targets)
    print("Test model output:", test_output[0].size())

    # Testing loss function
    loss = model.loss_function(*test_output)
    print("Loss from model:", loss.item())

model.to(device=device)

# Initializing optimizer
optimizer = torch.optim.AdamW(
    model.parameters(), 
    lr=tcfg.LEARNING_RATE, 
    eps=0.1     # High eps as per tensorflow notes: https://www.tensorflow.org/api_docs/python/tf/keras/optimizers/Adam
)
if not tcfg.REINITIALIZE_OPTIMIZER:
    if os.path.isdir(checkpoint_filepath) and tcfg.RESUME_LATEST:
        optimizer.load_state_dict(_optimizer_state_dict)
optimizer.zero_grad(set_to_none=True)

# Initializing loss function
criterion = model.loss_function

# Intiailzing checkpointing
checkpointing = Checkpointing(
            mode='min', 
            checkpoint_dir=checkpoint_filepath, 
            checkpoint_path='checkpoint.pth.tar'
        )

# Creating log file headers.
if not tcfg.DO_VALIDATION:
    if not _resume or os.path.exists(csv_log_path_perepoch):
        csv_logger(csv_log_path_perepoch, 'Epoch', 'Training Loss', 'Improvement', 'Datetime', reset=True)
    if tcfg.DEBUG_LOG_PER_ITERATION_LOSSES:
        if not tcfg.RESUME_LATEST or os.path.exists(csv_log_path_periterstep):    
            csv_logger(csv_log_path_periterstep, 'Epoch', 'Batch', 'Training Loss', 'Datetime', reset=True)
else:
    if not _resume or os.path.exists(csv_log_path_perepoch):
        csv_logger(csv_log_path_perepoch, 'Epoch', 'Training Loss', 'Validation Loss', 'Improvement', 'Datetime', reset=True)
    if tcfg.DEBUG_LOG_PER_ITERATION_LOSSES:
        if not tcfg.RESUME_LATEST or os.path.exists(csv_log_path_periterstep):    
            csv_logger(csv_log_path_periterstep, 'Epoch', 'Batch', 'Training Loss', 'Datetime', reset=True)

for epoch in range(_start_epoch, _start_epoch+tcfg.EPOCHS):
    epoch_start_time = time.time()

    running_loss = 0.0
    for iteration, (image_batch, target_batch) in enumerate(train_dataloader):
        # print(f"Epoch {epoch}: {(iteration+1)*batch_size}/{dataset_size}")
        start_time = time.time()

        model_output = model(image_batch.to(device), target_batch.to(device))

        if not tcfg.DO_GRADIENT_ACCUMULATION:
            loss = criterion(*model_output)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if tcfg.DEBUG_LOG_PER_ITERATION_LOSSES:
                csv_logger(csv_log_path_periterstep, epoch, iteration, loss.item(), datetime.datetime.now().strftime("%d/%m/%Y %H:%M:%S"),)
            print(f'Iteration {iteration+1} ({time.time() - start_time:.2f}s): Loss: {loss.item():{".4f" if loss.item() >= 1e-4 else ".4e"}}.')
        else:
            loss = criterion(*model_output)
            running_loss += loss.item()

            # Reweight losses.
            loss = loss / raw_batch_size
            # Backwards pass
            loss.backward()

            # After accumulation steps are reached, backpropagate gradients.
            if (iteration+1) % raw_batch_size == 0:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                if tcfg.DEBUG_LOG_PER_ITERATION_LOSSES:
                    csv_logger(csv_log_path_periterstep, epoch, iteration//raw_batch_size, loss.item(), datetime.datetime.now().strftime("%d/%m/%Y %H:%M:%S"),)
                print(f'Iteration {(iteration+1)/raw_batch_size} ({time.time() - start_time:.2f}s): Loss: {loss.item():{".4f" if loss.item() >= 1e-4 else ".4e"}}.')

    if tcfg.DO_VALIDATION:
        validation_loss = 0.0
        with torch.no_grad():
            for (valid_image_batch, valid_target_batch) in valid_dataloader:
                model_output = model(valid_image_batch.to(device), valid_target_batch.to(device))
                loss = criterion(*model_output)
                validation_loss += loss.item()
        is_best = checkpointing.check(validation_loss, epoch, model, optimizer)
        csv_logger(csv_log_path_perepoch, epoch, running_loss, validation_loss, is_best, datetime.datetime.now().strftime("%d/%m/%Y %H:%M:%S"),)
        print(f'Epoch {epoch+1} ({time.time() - epoch_start_time:.2f}s): Training loss: {running_loss:{".4f" if running_loss >= 1e-4 else ".4e"}}; Validation loss: {validation_loss:{".4f" if validation_loss >= 1e-4 else ".4e"}}{"*" if is_best else ""}')
    else:
        is_best = checkpointing.check(running_loss, epoch, model, optimizer)
        csv_logger(csv_log_path_perepoch, epoch, running_loss, is_best, datetime.datetime.now().strftime("%d/%m/%Y %H:%M:%S"),)
        print(f'Epoch {epoch+1} ({time.time() - epoch_start_time:.2f}s): Training loss: {running_loss:{".4f" if running_loss >= 1e-4 else ".4e"}}{"*" if is_best else ""}')

    if tcfg.DEBUG_BY_CHECKPOINTING:
        if (epoch+1) % tcfg.DEBUG_CHECKPOINT_EVERY == 0:
            torch.save(
                {
                    'epoch': epoch,
                    'state_dict': model.state_dict(),
                    'optimizer' : optimizer.state_dict(),
                }, 
                os.path.join(checkpoint_filepath, f"debug_checkpoint_{epoch+1}.pth.tar")
            )
            print(f"Saved debug checkpoint at epoch {epoch+1}.")