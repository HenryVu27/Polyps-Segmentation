import torch
# image params
IMG_SIZE = (256, 256)
CHANNELS = 3
MASK_CHANNELS = 1  # binary classification
BATCH_SIZE = 16

# pytorch params
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
NUM_WORKERS = 0  # number of workers for data loading
PIN_MEMORY = True  # faster data loading on NVDIA GPUs

# model params (for U-Net)
FILTERS_ROOT = 64
DEPTH = 4  # number of up/downsampling blocks
KERNEL_SIZE = 3  # conv kernel size
ACTIVATION = 'relu'
FINAL_ACTIVATION = 'sigmoid'

# training params
LEARNING_RATE = 1e-4
EPOCHS = 25

# data aug params
USE_AUGMENTATION = True
MAX_ROTATION = 20
HORIZONTAL_FLIP = True
VERTICAL_FLIP = False