# SOURCE: https://towardsdatascience.com/grad-cam-in-pytorch-use-of-forward-and-backward-hooks-7eba5e38d569

import os
import torch
import glob
import numpy as np

from torchvision.io.image import read_image
from torchvision.transforms.functional import normalize, resize, to_pil_image
from torchvision.models import resnet18
from torchcam.methods import SmoothGradCAMpp
from torchvision import transforms
from torchvision.transforms import ToTensor

import matplotlib.pyplot as plt
from PIL import Image
import PIL
import torch.nn.functional as F

from torchcam.utils import overlay_mask
from tqdm import tqdm

from matplotlib import colormaps

class GradCAMV2():
    def __init__(self, name_suffix) -> None:
        self.model = None
        self.name_suffix = name_suffix
        self.gradients = None
        self.activations = None

    def set_image_paths(self, img_path):
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

    def backward_hook(self, module, grad_input, grad_output):
        print('Backward hook running...')
        self.gradients = grad_output
        # In this case, we expect it to be torch.Size([batch size, 1024, 8, 8])
        print(f'Gradients size: {self.gradients[0].size()}') 
        # We need the 0 index because the tensor containing the gradients comes
        # inside a one element tuple.

    def forward_hook(self, module, args, output):
        print('Forward hook running...')
        self.activations = output
        # In this case, we expect it to be torch.Size([batch size, 1024, 8, 8])
        print(f'Activations size: {self.activations.size()}')

    def register_hooks(self):
        self.bck_hook = self.model.fc.register_full_backward_hook(self.backward_hook, prepend=False)
        self.fwd_hook = self.model.fc.register_forward_hook(self.forward_hook, prepend=False)

    def process_and_compute_data(self):
        # Loop through every image
        for idx in tqdm(range(len(self.image_paths))):
            image = Image.open(self.image_paths[idx]).convert('RGB')
            image_size = 64
            transform = transforms.Compose([
                                        transforms.Resize(image_size, antialias=True),
                                        transforms.CenterCrop(image_size),
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                    ])

            self.img_tensor = transform(image) # stores the tensor that represents the image
            # print(self.img_tensor.shape)

            # since we're feeding only one image, it is a 3d tensor (3, 256, 256). 
            # we need to unsqueeze to it has 4 dimensions (1, 3, 256, 256) as 
            # the model expects it to.

            # self.model(self.img_tensor.unsqueeze(0)).backward(torch.Tensor([2]))
            
            # here we did the forward and the backward pass in one line.

            # Compute CAM and Graph
            self.compute_grad_cam(self, idx)

    def compute_grad_cam(self, idx, save_to_dir: str = "./graphs/camV2_graphs"):
        # pool the gradients across the channels
        pooled_gradients = torch.mean(self.gradients[0], dim=[0, 2, 3])

        # weight the channels by corresponding gradients
        for i in range(self.activations.size()[1]):
            self.activations[:, i, :, :] *= pooled_gradients[i]

        # average the channels of the activations
        heatmap = torch.mean(self.activations, dim=1).squeeze()

        # relu on top of the heatmap
        heatmap = F.relu(heatmap)

        # normalize the heatmap
        heatmap /= torch.max(heatmap)

        # draw the heatmap
        # plt.matshow(heatmap.detach())

        # Create a figure and plot the first image
        fig, ax = plt.subplots()
        ax.axis('off') # removes the axis markers

        # First plot the original image
        ax.imshow(to_pil_image(self.img_tensor, mode='RGB'))

        # Resize the heatmap to the same size as the input image and defines
        # a resample algorithm for increasing image resolution
        # we need heatmap.detach() because it can't be converted to numpy array while
        # requiring gradients
        overlay = to_pil_image(heatmap.detach(), mode='F').resize((256,256), resample=PIL.Image.BICUBIC)

        # Apply any colormap you want
        cmap = colormaps['jet']
        overlay = (255 * cmap(np.asarray(overlay) ** 2)[:, :, :3]).astype(np.uint8)

        # Plot the heatmap on the same axes, 
        # but with alpha < 1 (this defines the transparency of the heatmap)
        ax.imshow(overlay, alpha=0.4, interpolation='nearest')

        # Show the plot
        # plt.show()
        plt.savefig(f'{save_to_dir}/CAM_{self.name_suffix}_{idx}.png')

    def remove_hooks(self):
        self.bck_hook.remove()
        self.fwd_hook.remove()


# SOURCE: https://medium.com/the-owl/gradcam-in-pytorch-7b700caa79e5
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

class GradCamModel(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.gradients = None
        self.tensorhook = []
        self.layerhook = []
        self.selected_out = None
        self.model = model
        
        # PRETRAINED MODEL
        self.pretrained = self.model
        self.layerhook.append(self.pretrained.layer4.register_forward_hook(self.forward_hook()))
        
        for p in self.pretrained.parameters():
            p.requires_grad = True
    
    def activations_hook(self,grad):
        self.gradients = grad

    def get_act_grads(self):
        return self.gradients

    def forward_hook(self):
        def hook(module, inp, out):
            self.selected_out = out
            self.tensorhook.append(out.register_hook(self.activations_hook))
        return hook

    def forward(self,x):
        out = self.pretrained(x)
        return out, self.selected_out


from torchcam.methods import GradCAM
def gradCamMethod(model, image_paths):

    # Loop through every image
    for idx in tqdm(range(len(image_paths))):
        image = Image.open(image_paths[idx]).convert('RGB')
        image_size = 64
        transform = transforms.Compose([
                                    transforms.Resize(image_size, antialias=True),
                                    transforms.CenterCrop(image_size),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                ])

        img_tensor = transform(image) # stores the tensor that represents the image
        # print(self.img_tensor.shape)

        cam = GradCAM(model, 'layer4')
        scores = model(img_tensor.unsqueeze(0))
        results = cam(class_idx=1, scores=scores)
        print(results)