import os
from PIL import Image
import torch
from tqdm import tqdm
import numpy as np
import glob
import cv2

import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image, ImageReadMode


# TODO: 
# Follow these steps from "Lecture 8: Training NNs Part III"

# 1. Read and normalize data (1 time; need to save and store as matrices)
# 2. Choose architecture (ResNet, CNN, VIM Transformer)
#   2.1. Disable regularization and check if loss is correct (log-likelihood)
#   2.2. Increase regularizaiton and check if loss goes up 
#   2.3. Run small test (small batch) - No regularizer, Test to see if model can overfit on small set
#   2.4. Start training - Adjust parameters (regularizer, learning rate, etc.)

# TODO:
# Try two versions of classifications: 
# - Keep non-Amherst cities separate
# - Combine all together as one label
# Create data loader (try tonight) - Class that return N datasets
# Pytorch custom dataloaders = train on custom datasets - custom image classifier 
# (https://dilithjay.com/blog/custom-image-classifier-with-pytorch/)

# Loss Function
# Binary Cross-Entropy Loss for two classes/labels (Amherst vs. Non-Amherst) 
# Cross-Entropy Loss for multiple classes/labels (Amherst vs. NY vs. etc.)


class CustomDataset(Dataset):
    def __init__(self, root_dir, transform):
        self.transform = transform

        # Check if root_dir exists
        if not os.path.exists(root_dir):
            raise Exception(f"root_dir {root_dir} does not exist")
        
        # Get list of all unique city labels
        # This works for cases for binary and non-binary labels
        self.label_dirs = glob.glob(os.path.join(root_dir, "*", ""))

        # Create class_lbl
        class_set = set()
        for path in self.label_dirs:
            class_set.add(os.path.basename(os.path.dirname(path)))
        self.class_lbl = { cls: i for i, cls in enumerate(sorted(list(class_set)))}
        
        # Extract all images
        self.image_paths = []
        for ext in ['png', 'jpg']:
            glob_path = os.path.join(root_dir, '**/', f'*.{ext}')
            self.image_paths += glob.glob(glob_path, recursive=True)
        
        # class_set = set()
        # for path in self.image_paths:
        #     class_set.add(os.path.dirname(path))
        # self.class_lbl = { cls: i for i, cls in enumerate(sorted(list(class_set)))}

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Read image of from image_paths given index
        img_dir = self.image_paths[idx]
        img = read_image(img_dir, ImageReadMode.RGB).float()
        # Ex: img_dir = '/Users/Preston/CS-682/Final_Project/dataset/Binary_Dataset/Non-Amherst/Zurich_dataset/png-ZuBuD/object0071.view05.png'
        
        # Identify label for that index
        cls = None
        for dir in self.label_dirs:
            # If img_dir is a subdirectory of dir, get basename
            # Ex: self.label_dirs = '/Users/Preston/CS-682/Final_Project/dataset/Binary_Dataset/Non-Amherst/'
            if img_dir.startswith(os.path.abspath(dir)+os.sep):
                if dir.endswith('/'):
                    dir = dir[:-1]
                cls = os.path.basename(dir)
                # Ex: cls = 'Non-Amherst'
                break

        if cls is None or len(cls) <= 0:
            raise Exception(f"img_dir {img_dir} not found in self.label_dirs")
        
        # Get label
        label = self.class_lbl[cls]
        # Ex: label = 1 (self.class_lbl = {"Amherst": 0, "Non-Amherst": 1})

        return self.transform(img), torch.tensor(label)
    

def data_loader(dataset_dir: str):
    """
    Extract images of convert into dataset for training
    :param dataset_dir: Location of where images are
    :param save_to_dir: Location of where to save numpy array dataset (default is "dataset" folder)
    Note: Use np.load('data.npy') to extract data
    """
    print("\n------------------ RUNNING DATA LOADER ------------------")

    # Define transformer
    # Calling tranform(image) would return an image which is an augmented version of the input image.
    # TODO: Data Augmentation in transform doc
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.RandomRotation(degrees=20),
        transforms.RandomRotation(degrees=40),
        # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        # TODO: Try "trimming" out background sky (center)
    ])

    # Instantiate dataset object
    dataset = CustomDataset(root_dir=f'{dataset_dir}/', transform=transform)

    # Create ratio for splitting dataset (train, val)
    splits = [0.8, 0.2]
    split_sizes = []
    for sp in splits[:-1]:
        split_sizes.append(int(sp * len(dataset)))
    split_sizes.append(len(dataset) - sum(split_sizes))

    # Use random_split to split dataset
    train_set, val_set = torch.utils.data.random_split(dataset, split_sizes)

    # Get list of validation images dataset (Needed to run CAM on)
    val_img_list = [dataset.image_paths[i] for i in tuple(val_set.indices)]

    # Create dataloaders
    dataloaders = {
        "train": DataLoader(train_set, batch_size=16, shuffle=True),
        "val": DataLoader(val_set, batch_size=16, shuffle=False),
    }

    print("Data loading successful!")
    return dataloaders, dataset, val_img_list


def extract_dataset(data_loader: DataLoader):
    """
    Extract images from DataLoader (can be used for either training or validation data)
    """
    dataset_list = []
    for images, labels in data_loader:
        dataset_list.append(images)
    print("Finished extracting dataloader")

    return dataset_list