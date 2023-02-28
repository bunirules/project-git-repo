# type: ignore
import torch
import numpy as np
import glob as glob
from torch.utils.data import DataLoader, Dataset
from PIL import Image
TRAIN_BATCH_SIZE = 128
TEST_BATCH_SIZE = 8

# The SRCNN dataset module.
class SRCNNDataset(Dataset):
    def __init__(self, image_paths, label_paths):
        self.all_image_paths = glob.glob(f"{image_paths}/*")
        self.all_label_paths = glob.glob(f"{label_paths}/*")
    def __len__(self):
        return (len(self.all_image_paths))
    def __getitem__(self, index):
        image = Image.open(self.all_image_paths[index])
        label = Image.open(self.all_label_paths[index])
        image = np.array(image, dtype=np.float32)
        label = np.array(label, dtype=np.float32)
        image = image.reshape([1,image.shape[0],image.shape[1]])
        label = label.reshape([1,label.shape[0],label.shape[1]])
        if 256 <= np.max(image) <= 65535:
            image //= 256.
        if 1 < np.max(image) <= 255:
            image /= 255.
        if 256 <= np.max(label) <= 65535:
            label //= 256.
        if 1 < np.max(label) <= 255:
            label /= 255.
        return (
            torch.tensor(image, dtype=torch.float),
            torch.tensor(label, dtype=torch.float)
        )
# Prepare the datasets.
def get_datasets(
    train_image_paths, train_label_paths,
    valid_image_path, valid_label_paths
):
    dataset_train = SRCNNDataset(
        train_image_paths, train_label_paths
    )
    dataset_valid = SRCNNDataset(
        valid_image_path, valid_label_paths
    )
    return dataset_train, dataset_valid
# Prepare the data loaders
def get_dataloaders(dataset_train, dataset_valid):
    train_loader = DataLoader(
        dataset_train, 
        batch_size=TRAIN_BATCH_SIZE,
        shuffle=True
    )
    valid_loader = DataLoader(
        dataset_valid, 
        batch_size=TEST_BATCH_SIZE,
        shuffle=False
    )
    return train_loader, valid_loader