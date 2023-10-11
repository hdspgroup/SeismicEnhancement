import os
from dataclasses import dataclass
from .utils import get_default_device

@dataclass
class BaseConfig:
    DEVICE = get_default_device()
    DATASET = "Seismic" #  "MNIST", "Cifar-10", "Cifar-100", "Flowers", "Seismic"
    
    # For logging inferece images and saving checkpoints.
    root_log_dir = os.path.join("Logs_Checkpoints", "Inference")
    root_checkpoint_dir = os.path.join("Logs_Checkpoints", "checkpoints")

    # Current log and checkpoint directory.
    log_dir = "version_0"
    checkpoint_dir = "version_0"

@dataclass
class TrainingConfig:
    TIMESTEPS = 1000 # Define number of diffusion timesteps
    IMG_SHAPE = (1, 32, 32) if BaseConfig.DATASET == "MNIST" else (1, 128, 128) 
    NUM_EPOCHS = 10000
    BATCH_SIZE = 12
    LR = 2e-4
    NUM_WORKERS = 6
    CONDITIONAL = False

@dataclass
class ModelConfig:
    BASE_CH = 32  # 64, 128, 256, 512
    BASE_CH_MULT = (1, 2, 4, 8) # 32, 16, 8, 4 
    APPLY_ATTENTION = (False, False, True, False)
    DROPOUT_RATE = 0.1
    TIME_EMB_MULT = 2 # 128
