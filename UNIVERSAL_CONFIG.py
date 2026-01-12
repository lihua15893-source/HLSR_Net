import torch

class UNIVERSAL_CONFIG:
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    OUTPUT_DIR = 'save_model'
    IMAGE_SIZE = 512
    NUM_CLASSES = 2
    USE_AUGMENTATION = True
    BILINEAR = True
    USE_BATCHNORM = True
    DROPOUT = 0.0
    BASE_CHANNEL = 64
    USE_PRETRAINED = False
    EPOCHS = 100
    BATCH_SIZE = 16
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-4
    LR_SCHEDULER = 'reduce'
    LR_STEP_SIZE = 30
    LR_GAMMA = 0.8
    LR_MIN = 1e-7
    EARLY_STOPPING = True
    PATIENCE = 50
    LOSS_TYPE = 'ce'
    DICE_WEIGHT = 0.5
    CE_WEIGHT = 0.5
    BOUNDARY_WEIGHT = 0.2
    OPTIMIZER = 'adamw'
    MOMENTUM = 0.9
    BETAS = (0.9, 0.999)
    LOG_FREQ = 100
    VAL_FREQ = 1
    CHECKPOINT_POLICY = 'best'