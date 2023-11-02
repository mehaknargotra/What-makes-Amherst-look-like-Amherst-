from PIL import Image
from tqdm import tqdm
import numpy as np
import glob
import cv2

# TODO: 
# Follow these steps from "Lecture 8: Training NNs Part III"

# 1. Read and normalize data (1 time; need to save and store as matrices)
# 2. Choose architecture (ResNet, CNN, VIM Transformer)
#   2.1. Disable regularization and check if loss is correct (log-likelihood)
#   2.2. Increase regularizaiton and check if loss goes up 
#   2.3. Run small test (small batch) - No regularizer, Test to see if model can overfit on small set
#   2.4. Start training - Adjust parameters (regularizer, learning rate, etc.)


def create_dataset(dataset_dir: str, dataset_type: str, save_to_name: str, save_to_dir: str="./dataset"):
    """
    Extract images of type dataset_type from dataset, convert to numpy array, and save/store as .npy file
    :param dataset_dir: Location of where images are
    :param dataset_type: Filetype to search for
    :param save_to_name: Name of dataset file to save as
    :param save_to_dir: Location of where to save numpy array dataset (default is "dataset" folder)
    Note: Use np.load('data.npy') to extract data
    """

    # Read data
    images = glob.glob(f"{dataset_dir}/**/*{dataset_type}", recursive=True) 
    print(f"Images found: {len(images)}")

    # Loop through all images found
    print("Loading data...")
    img_data_list = []
    for img in tqdm(images):
        # Get image
        get_img = cv2.imread(img)

        # TODO: Resize images to be of consistent shape
        # res = cv2.resize(img, dsize=(54, 140), interpolation=cv2.INTER_CUBIC)
        
        # Convert image to numpy array
        data = np.asarray(get_img) # Should be a shape of (H, W, C)

        # Reshape to (1, H, W, C) for concatenation
        data = np.reshape(data, (1, data.shape[0], data.shape[1], data.shape[2])) 
        
        # Add to list
        img_data_list.append(data)
    print("Data Collected")

    # Concatenate list of image data arrays into one whole numpy array
    if len(img_data_list) == 0:
        raise Exception("No images found/converted to np.array")
    print(f"Image Shape: {img_data_list[0].shape}")
    img_dataset = np.concatenate(img_data_list, axis=0)
    print(f"Numpy Array Dataset Created: {img_dataset.shape}") # (N, H, W, C)
    
    # Save to npy file
    save_to_path = f'{save_to_dir}/{save_to_name}.npy'
    np.save(save_to_path, img_dataset)
    print(f"Data saved to: {save_to_path}")


create_dataset(
    dataset_dir = "/Users/Preston/CS-682/Final_Project/dataset/Amherst",
    dataset_type = ".jpg",
    save_to_name = "Amherst"
    )