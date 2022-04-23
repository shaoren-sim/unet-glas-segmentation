class TrainingConfig():
    # Default experiment name. Model logs and results will be saved in "checkpointing/EXPERIMENT_NAME"
    EXPERIMENT_NAME = "experiment"

    # Can be "cpu" or "cuda:{ind}", where {ind} is the index of the GPU used for training.
    DEVICE = "cuda:0"

    # How many epochs to train for.
    EPOCHS = 4000
    BATCH_SIZE = 16     # If batch size is larger than 1, gradient accumulation will be done.
    LEARNING_RATE = 5e-3        # Default is 5e-3

    # Whether or not to do validation on test data.
    DO_VALIDATION = True   

    # Whether to do augmentations, 
    # if False, Dataset will use data files as is, with no augmentations.
    # if True, data will be augmented with spatial deformations.
    DO_AUGMENTATIONS = True     # Default is True

    # Whether to use Gradient Accumulation, used when batch size exceeds available memory.
    # https://ai.stackexchange.com/questions/21972/what-is-the-relationship-between-gradient-accumulation-and-batch-size
    # If DO_GRADIENT_ACCUMULATION==True, each batch will take BATCH_SIZE // GRADIENT_ACCUMULATION_STEPS samples.
    # Example: For BATCH_SIZE==4096, GRADIENT_ACCUMULATION_STEPS==128, each sub-batch will include 32 samples.
    # If BATCH_SIZE==4096 with DO_GRADIENT_ACCUMULATION==False, a sub-batch will include 4096 samples.
    DO_GRADIENT_ACCUMULATION = True

    # Checkpointing folders
    DO_CHECKPOINTING = True
    CHECKPOINTING_DIRECTORY = "checkpoints"

    # Whether to resume from the latest checkpoint.
    RESUME_LATEST = True
    REINITIALIZE_OPTIMIZER = False      # Optimizer stats can be restarted if training parameters change (i.e. learning rate change, batch size change.)

    # Debug options
    DEBUG_BY_CHECKPOINTING = True      # If true, saves debugging checkpoints every N epochs
    DEBUG_CHECKPOINT_EVERY = 200       # How often to save debug checkpoints
    DEBUG_LOG_PER_ITERATION_LOSSES = True    # Whether to log loss at every iteration. Results in massive log files, but can observe if losses spike mid training.

class AugmentationConfig:
    # Parameters for elastic deformation
    ELASTIC_DEFORM_SIGMA_MIN = 10
    ELASTIC_DEFORM_SIGMA_MAX = 50
    ELASTIC_DEFORM_ALPHA_MIN = 100
    ELASTIC_DEFORM_ALPHA_MAX = 300