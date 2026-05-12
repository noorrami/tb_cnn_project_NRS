import torch

# Dataset path
DATA_DIR = '/kaggle/input/tuberculosis-tb-chest-xray-dataset/TB_Chest_Radiography_Database'

# Hyperparameters
BATCH_SIZE = 32
LEARNING_RATE = 0.001
NUM_EPOCHS = 10
VAL_SPLIT = 0.2
IMAGE_SIZE = 224

# Technical settings
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')