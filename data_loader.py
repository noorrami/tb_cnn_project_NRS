import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from config import DATA_DIR, BATCH_SIZE, VAL_SPLIT, IMAGE_SIZE

def get_transforms():
    """Preparing image conversions"""
    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    return transform

def load_data():
    """Loading and splitting data"""
    full_data = datasets.ImageFolder(DATA_DIR)
    transform = get_transforms()
    full_data.transform = transform

    train_size = int((1 - VAL_SPLIT) * len(full_data))
    val_size = len(full_data) - train_size
    train_dataset, val_dataset = random_split(full_data, [train_size, val_size])

    # We guarantee that the conversions apply to both parts.
    train_dataset.dataset.transform = transform
    val_dataset.dataset.transform = transform

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    return train_loader, val_loader