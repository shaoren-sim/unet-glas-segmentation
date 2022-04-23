# To run tests, run this shell script from the Python environment with all requirements installed.
# torchinfo must be installed to run the tests.

# These tests check if the models can be forward-passed through successfully.
python -m models.encoders.resnet
python -m models.decoders.resnet

# These tests ensure that the encoder and decoder backbones can be successfuly linked for UNet and FCN configurations
python -m models.fcn
python -m models.unet

# These tests ensure that augmentations are properly executed (Probabilities are maximized).
echo "Testing dataloading (plots will appear)"
python -m utilities.data

echo "Testing augmentations (plots will appear)"
python -m utilities.preprocessing.augments