import numpy as np
import os
import torch
import glob

from torchvision.io.image import read_image
from torchvision.transforms.functional import normalize, resize, to_pil_image
from torchvision.models import resnet18
from torchcam.methods import SmoothGradCAMpp
from torchvision import transforms

import matplotlib.pyplot as plt
from torchcam.utils import overlay_mask
from tqdm import tqdm


# TODO: Class Activation Mapping; Need to research and implement/test/tune
# Will be running multiple combinations between CAM and the classifiers

# Links/References:
# https://github.com/topics/class-activation-map
# *** https://medium.com/intelligentmachines/implementation-of-class-activation-map-cam-with-pytorch-c32f7e414923
# *** https://github.com/frgfm/torch-cam
# *** https://leslietj.github.io/2020/07/15/PyTorch-Implementation-of-Class-Activation-Map-CAM/
# https://github.com/frgfm/torch-cam/discussions/132


class ClassActivationMap():
    def __init__(self) -> None:
        self.model = None
        self.activation_map = None
        self.image_paths = []
        self.activation_map_list = []
        self.dataset = []
        
    def set_image_paths(self, img_path):
        self.image_paths = img_path
    
    def extract_imgs(self, dir: str, num_imgs: int = -1):
        """
        Extract images to test on CAM
        :param dir: directory to extract images from (specifically looking for Amherst)
        :param num_imgs: number of image dirs to extract into a list (If -1, extract all)
        """

        # Check that dir exists
        if not os.path.exists(dir):
            raise Exception(f"Directory {dir} not found")
        
        # Extract list of image dirs
        print("Extracting images ...")
        self.image_paths = []
        for ext in ['png', 'jpg']:
            glob_path = f"{dir}/**/*.{ext}"
            self.image_paths += glob.glob(glob_path, recursive=True)

        # Extract num_imgs if specified
        if num_imgs > 0:
            self.image_paths = self.image_paths[-num_imgs:]

    def extract_model(self, file_name: str, dir: str = "./models", file_type: str = ".h5"):
        """
        Extract model saved inside "model" folder
        :param file_name: name of file where model is saved
        :param dir: directory where to find file (default: "model" folder)
        :param file_type: type of file to extract form (default: .h5)
        """
        print("Extracting model ...")
        
        # Form path to get to saved model
        model_path = f"{dir}/{file_name}{file_type}"

        # Check if model exists
        if not os.path.exists(model_path):
            raise Exception(f"File {file_name} not found in {dir}")

        # Extract file
        self.model = torch.load(model_path)['model']
        self.model.eval()

    def set_image_batch(self, dataset_list: list, idx_range: list = None):
        """
        :param dataset_list: list of validation dataset from dataloader
        Note: Dataset is of shape (N, C, H, W), where N = Number of Batches
        :param idx_range: list of index range of where to start and stop with accessing dataset
            If None, then do all
        """
        # Get batches of
        if idx_range is not None:
            batch_dataset = dataset_list[idx_range[0]:idx_range[1]]
        else:
            batch_dataset = dataset_list
        
        dataset_list = []
        for batch in batch_dataset:
            bat_split = list(torch.split(batch, 1, dim=0))
            # print("bat_split:", bat_split)

            dataset_list = [*dataset_list, *bat_split] 
            # dataset_list.append(list(torch.squeeze(torch.split(batch, 1, dim=0))))

        # print("dataset_list:", dataset_list)
        self.dataset = dataset_list

    def run(self):
        """
        Run CAM on model
        """
        print("\n------------------ RUNNING CAM ------------------")

        if len(self.image_paths) > 0:
            # Loop through for every image in list
            for img_path in tqdm(self.image_paths):

                # Extract/read image
                img = read_image(img_path)

                # Create transformer
                test_transforms = transforms.Compose([
                    transforms.Resize(224),
                    # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                    ])

                # Preprocess it for your chosen model
                # input_tensor = normalize(resize(img, (224, 224)) / 255., [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                # img_arr = img.detach().cpu().numpy()
                input_tensor = test_transforms(img.type(torch.FloatTensor))
                
                with SmoothGradCAMpp(self.model) as cam_extractor:
                    # Preprocess your data and feed it to the model
                    out = self.model(input_tensor.unsqueeze(0))
                    
                    # Retrieve the CAM by passing the class index and the model output
                    self.activation_map_list.append(cam_extractor(out.squeeze(0).argmax().item(), out))

    def graph(self, file_suffix: str, save_to_dir: str = "./graphs/cam_graphs"):
        """
        Show heatmap overlaid on top of all images in list
        :param file_suffix: string suffix for graph file names
        :param save_to_dir: location of where to save graphs in
        """
        # Create directory/folder for graph data
        save_to_dir = os.path.join(save_to_dir, f"CAM_{file_suffix}")
        if not os.path.exists(save_to_dir):
            os.mkdir(save_to_dir)

        print("Graphing ...")

        # Loop through for every image in list
        for idx in tqdm(range(len(self.image_paths))):

            # Get image
            img = read_image(self.image_paths[idx])

            # Get activation map
            activation_map = self.activation_map_list[idx]

            # Resize the CAM and overlay it
            result = overlay_mask(to_pil_image(img), to_pil_image(activation_map[0].squeeze(0), mode='F'), alpha=0.5)
        
            # Save images
            plt.imshow(result)
            plt.axis('off')
            plt.tight_layout()
            plt.savefig(f'{save_to_dir}/CAM_{file_suffix}_{idx}.png')
