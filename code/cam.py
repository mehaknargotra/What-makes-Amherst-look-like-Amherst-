import os
import torch

from torchvision.io.image import read_image
from torchvision.transforms.functional import to_pil_image
from torchcam.methods import SmoothGradCAMpp, GradCAMpp, CAM
from torchvision import transforms

import matplotlib.pyplot as plt
from torchcam.utils import overlay_mask
from tqdm import tqdm


""" Class Activation Mapping """


# Links/References:
# https://github.com/topics/class-activation-map
# https://medium.com/intelligentmachines/implementation-of-class-activation-map-cam-with-pytorch-c32f7e414923
# https://github.com/frgfm/torch-cam
# https://leslietj.github.io/2020/07/15/PyTorch-Implementation-of-Class-Activation-Map-CAM/
# https://github.com/frgfm/torch-cam/discussions/132


class ClassActivationMap():
    def __init__(self) -> None:
        self.model = None
        self.activation_map = None
        self.image_paths = []
        self.activation_map_list = []
        self.dataset = []
        
    def set_image_paths(self, img_path):
        """
        Save image_paths (Validation image paths to run CAMs on)
        :param img_path: List of paths to validation images
        """
        self.image_paths = img_path
    
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

    def run(self, cam_type=1):
        """
        Run CAM on model (Smooth Grad CAM)

        :param cam_type: Type of CAM to run 
            1: SmoothGradCAMpp
            2: GradCAMpp
            3: CAM
        """
        print("\n------------------ RUNNING CAM ------------------")

        if len(self.image_paths) > 0:
            # Loop through for every image in list
            for img_path in tqdm(self.image_paths):

                # Extract/read image
                img = read_image(img_path)

                # Create transformer
                test_transforms = transforms.Compose([
                    transforms.Resize(256),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])

                # Preprocess it for your chosen model
                input_tensor = test_transforms(img.type(torch.FloatTensor))

                # Switch between gpu or cpu
                device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
                input_tensor = input_tensor.to(device)
                
                if cam_type == 1:
                    with SmoothGradCAMpp(self.model, 'layer4') as cam_extractor:
                        # Preprocess data and feed into model
                        out = self.model(input_tensor.unsqueeze(0))
                        
                        # Retrieve CAM with class index and model output
                        self.activation_map_list.append(cam_extractor(out.squeeze(0).argmax().item(), out))
                elif cam_type == 2:
                    with GradCAMpp(self.model, 'layer4') as cam_extractor:
                        out = self.model(input_tensor.unsqueeze(0))
                        self.activation_map_list.append(cam_extractor(out.squeeze(0).argmax().item(), out))
                else:
                    with CAM(self.model, 'layer4') as cam_extractor:
                        out = self.model(input_tensor.unsqueeze(0))
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
            try:  
                os.mkdir(save_to_dir)  
            except OSError as error:  
                print(error)

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
            plt.clf()
