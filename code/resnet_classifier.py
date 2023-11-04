import numpy as np
from torchvision.models import resnet50, ResNet50_Weights
from tqdm import tqdm
import torch


# TODO: Need to get/copy ResNet classifier
# Links: 
# https://www.kaggle.com/code/toygarr/resnet-implementation-for-image-classification
# https://datagen.tech/guides/computer-vision/resnet-50/#
# https://pyimagesearch.com/2020/04/27/fine-tuning-resnet-with-keras-tensorflow-and-deep-learning/


# Simple Fully-Connected model example
# NOTE: We will be applied to our model.fc
class FCModel(torch.nn.Module):
    def __init__(self, input_dim, num_classes=2):
        super(FCModel, self).__init__()

        self.linear1 = torch.nn.Linear(input_dim, 256)
        self.relu = torch.nn.ReLU()
        if num_classes == 2:
            # self.linear2 = torch.nn.Linear(256, 1)
            self.linear2 = torch.nn.Linear(256, num_classes)
        else:
            self.linear2 = torch.nn.Linear(256, num_classes)
        self.softmax = torch.nn.Sigmoid()

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        x = self.softmax(x)
        return x


class ResNetClassifier():
    
    def __init__(self, 
                 data_loaders: dict,
                 num_labels: int, 
                 optimizer: str = "adam", 
                 learning_rate: float = 0.0001, 
                 epoches: int = 30) -> None:
        
        # Get/init train/validation data_loaders dict from data_loader()
        self.dataloaders = data_loaders

        # Save number of labels/classes
        self.num_labels = num_labels

        # Initialize the model with its pretrained weights (init weights?)
        # TODO: Fine-tuning ResNet50 on custom dataset
        self.model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)

        # Add FC Layer
        self.model.fc = FCModel(input_dim=2048, num_classes=self.num_labels)

        # Use GPU if available
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)

        # You may also want to freeze parameters of the pretrained network rather than 
        # fine-tune the entire network. 
        # To do this, we can simply set the requires_grad attribute of all the parameters 
        # (except the new parameters) to False. 
        # This prevents PyTorch from calculating the gradients for those parameters and thus, 
        # doesn’t update them.
        for param in self.model.parameters():
            param.requires_grad = False

        for param in self.model.fc.parameters():
            param.requires_grad = True

        # Define loss function
        if self.num_labels == 2:
            # Binary Cross Entropy Loss for 2 classes (Amherst vs. Non-Amherst)
            self.criterion = torch.nn.BCELoss()
        else:
            # Cross Entropy Loss for multiple classes (Amherst vs. City1 vs. City2 vs. ...)
            # TODO: Do this later
            self.criterion = torch.nn.CrossEntropyLoss()

        # Define optimizer
        optimizer = optimizer.lower()
        if optimizer.lower() == "adam":
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        elif optimizer.lower() == "sgd":
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=learning_rate, momentum=0.9)
        else:
            raise Exception(f"Optimizer {optimizer} does not exist")

        # Create metrics for training
        self.metrics = {
            'train': {
                'loss': [], 'accuracy': []
            },
            'val': {
                'loss': [], 'accuracy': []
            },
        }

        # Set number of epoches for training
        self.epoches = epoches

    def train(self) -> None:
        """
        Training code from following link:
        https://dilithjay.com/blog/custom-image-classifier-with-pytorch/
        """
        
        for epoch in range(self.epoches):
            ep_metrics = {
                'train': {'loss': 0, 'accuracy': 0, 'count': 0},
                'val': {'loss': 0, 'accuracy': 0, 'count': 0},
            }
            print(f'Epoch {epoch}')

            for phase in ['train', 'val']:
                print(f'-------- {phase} --------')
                for images, labels in tqdm(self.dataloaders[phase]):
                    self.optimizer.zero_grad()

                    with torch.set_grad_enabled(phase == 'train'):
                        output = self.model(images.to(self.device))
                        ohe_label = torch.nn.functional.one_hot(labels, num_classes=self.num_labels)

                        loss = self.criterion(output, ohe_label.float().to(self.device))

                        correct_preds = labels.to(self.device) == torch.argmax(output, dim=1)
                        accuracy = (correct_preds).sum()/len(labels)

                    if phase == 'train':
                        loss.backward()
                        self.optimizer.step()

                    ep_metrics[phase]['loss'] += loss.item()
                    ep_metrics[phase]['accuracy'] += accuracy.item()
                    ep_metrics[phase]['count'] += 1
            
                ep_loss = ep_metrics[phase]['loss']/ep_metrics[phase]['count']
                ep_accuracy = ep_metrics[phase]['accuracy']/ep_metrics[phase]['count']

                print(f'Loss: {ep_loss}, Accuracy: {ep_accuracy}\n')

                self.metrics[phase]['loss'].append(ep_loss)
                self.metrics[phase]['accuracy'].append(ep_accuracy)


# Old Function: Ignore
def loss_function(num_labels):
    # Initialize the model with its pretrained weights (init weights?)
    model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)

    # Use GPU if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    # You may also want to freeze parameters of the pretrained network rather than 
    # fine-tune the entire network. 
    # To do this, we can simply set the requires_grad attribute of all the parameters 
    # (except the new parameters) to False. 
    # This prevents PyTorch from calculating the gradients for those parameters and thus, 
    # doesn’t update them.
    for param in model.parameters():
        param.requires_grad = False

    for param in model.fc.parameters():
        param.requires_grad = True

    # Define loss function
    if num_labels == 2:
        # Binary Cross Entropy Loss for 2 classes (Amherst vs. Non-Amherst)
        criterion = torch.nn.BCELoss()
    else:
        # Cross Entropy Loss for multiple classes (Amherst vs. City1 vs. City2 vs. ...)
        # TODO: Do this later
        criterion = torch.nn.CrossEntropyLoss()

    # Define optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    # Training classifier model
    metrics = {
        'train': {
            'loss': [], 'accuracy': []
        },
        'val': {
            'loss': [], 'accuracy': []
        },
    }

    return 0


