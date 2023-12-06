from torchvision.models import resnet50, ResNet50_Weights
from tqdm import tqdm
import torch
import os
import matplotlib.pyplot as plt


""" ResNet Classifier """

# REFS: 
# https://www.kaggle.com/code/toygarr/resnet-implementation-for-image-classification
# https://datagen.tech/guides/computer-vision/resnet-50/#
# https://pyimagesearch.com/2020/04/27/fine-tuning-resnet-with-keras-tensorflow-and-deep-learning/


# Simple Fully-Connected model applied to our model.fc
class FCModel(torch.nn.Module):
    def __init__(self, input_dim, num_classes=2):
        super(FCModel, self).__init__()

        self.linear1 = torch.nn.Linear(input_dim, 256)
        self.relu = torch.nn.ReLU()
        self.linear2 = torch.nn.Linear(256, num_classes)
        self.softmax = torch.nn.Sigmoid()

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        x = self.softmax(x)
        return x


# Class for ResNet Training
class ResNetClassifier():
    def __init__(self, 
                 data_loaders: dict,
                 num_labels: int, 
                 optimizer: str = "adam", 
                 learning_rate: float = 0.0001, 
                 epochs: int = 30) -> None:
        
        # Get/init train/validation data_loaders dict from data_loader()
        self.dataloaders = data_loaders

        # Save number of labels/classes
        self.num_labels = num_labels

        # Initialize the model with its pretrained weights
        self.model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)

        # Add FC Layer
        self.model.fc = FCModel(input_dim=2048, num_classes=self.num_labels)

        # Use GPU if available
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)

        for param in self.model.parameters():
            # Set to true to test out cam.py
            param.requires_grad = True

        for param in self.model.fc.parameters():
            param.requires_grad = True

        # Define loss function
        if self.num_labels == 2:
            # Binary Cross Entropy Loss for 2 classes (Amherst vs. Non-Amherst)
            self.criterion = torch.nn.BCELoss()
        else:
            # Cross Entropy Loss for multiple classes (Amherst vs. City1 vs. City2 vs. ...)
            self.criterion = torch.nn.CrossEntropyLoss()

        # Define optimizer
        optimizer = optimizer.lower()
        if optimizer.lower() == "adam":
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        elif optimizer.lower() == "sgd":
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=learning_rate, momentum=0.9)
        else:
            raise Exception(f"Optimizer {optimizer} does not exist")

        # Create training and validation metrics for training (loss and accuracy)
        self.metrics = {
            'train': {
                'loss': [], 'accuracy': []
            },
            'val': {
                'loss': [], 'accuracy': []
            },
        }

        # Set number of epochs for training
        self.epochs = epochs

    def train(self) -> None:
        """
        Train ResNet on dataset
        REF: https://dilithjay.com/blog/custom-image-classifier-with-pytorch/
        """
        print("\n------------------ BEGIN TRAINING ------------------")
        
        # Loop over all the epochs
        for epoch in range(self.epochs):
            
            # Define a dictionary, similar to the one we defined earlier, to keep track of the metrics for the current epoch.
            ep_metrics = {
                'train': {'loss': 0, 'accuracy': 0, 'count': 0},
                'val': {'loss': 0, 'accuracy': 0, 'count': 0},
            }
            print(f'Epoch {epoch}')

            # Do both training and validation for each epoch.
            for phase in ['train', 'val']:
                print(f'------------ {phase} ------------')
                
                # Loop over the batches of data for images and its labels
                for images, labels in tqdm(self.dataloaders[phase]):
                    
                    # Reset optimizer’s gradient
                    self.optimizer.zero_grad()

                    # If in training phase, keep track of gradients 
                    # If not, don’t keep track of gradient since it saves compute power and memory
                    with torch.set_grad_enabled(phase == 'train'):
                        
                        # Run images through model and get predicted output
                        output = self.model(images.to(self.device))
                        
                        # Turn the labels into one-hot encoded vectors
                        ohe_label = torch.nn.functional.one_hot(labels, num_classes=self.num_labels)

                        # Use one-hot encoded labels to calculate loss
                        loss = self.criterion(output, ohe_label.float().to(self.device))

                        # Use argmax to get predicted labels and use them with ground truth labels to calculate accuracy
                        correct_preds = labels.to(self.device) == torch.argmax(output, dim=1)
                        accuracy = (correct_preds).sum()/len(labels)

                    if phase == 'train':
                        # Backpropagate loss and calculate gradients
                        loss.backward()

                        # Update weights
                        self.optimizer.step()

                    # Keep track of total accuracy, total loss, and batch count, 
                    # Used to calculate accuracy and loss for entire epoch.
                    ep_metrics[phase]['loss'] += loss.item()
                    ep_metrics[phase]['accuracy'] += accuracy.item()
                    ep_metrics[phase]['count'] += 1
            
                # Calculate epoch loss and accuracy, and update metrics dictionary.
                ep_loss = ep_metrics[phase]['loss']/ep_metrics[phase]['count']
                ep_accuracy = ep_metrics[phase]['accuracy']/ep_metrics[phase]['count']

                # Print and log loss and accuracy
                print(f'Loss: {ep_loss}, Accuracy: {ep_accuracy}\n')
                self.metrics[phase]['loss'].append(ep_loss)
                self.metrics[phase]['accuracy'].append(ep_accuracy)

    def graph(self, suffix: str, save_to_dir: str = "./graphs/resnet_graphs"):
        """
        Show graphs for Loss and Accuracy for both training and validation
        :param suffix: detail of suffix for file name
        :param save_to_dir: location to save Training/Validation Loss/Accuracy graphs (default: resnet_graphs folder)
        """

        # Create save_to_dir if it doesn't exist
        if not os.path.exists(save_to_dir):
            try:  
                os.mkdir(save_to_dir)  
            except OSError as error:  
                print(error)

        # Get epochs
        epochs_range = range(0, self.epochs)

        # Show graph of training and validation accuracies
        train_acc = self.metrics['train']['accuracy']
        val_acc = self.metrics['val']['accuracy']

        # Create accuracy graph
        plt.figure()
        plt.plot(epochs_range, train_acc, 'g', label="Training Accuracy")
        plt.plot(epochs_range, val_acc, 'b', label="Validation Accuracy")
        plt.title(f'Training and Validation Accuracy ({suffix})')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        # plt.show()
        plt.savefig(f'{save_to_dir}/ResNet_Accuracy_{suffix}.png')
        plt.close()

        # Show graph of training and validation accuracies
        train_loss = self.metrics['train']['loss']
        val_loss = self.metrics['val']['loss']

        # Create accuracy graph
        plt.figure()
        plt.plot(epochs_range, train_loss, 'g', label="Training Loss")
        plt.plot(epochs_range, val_loss, 'b', label="Validation Loss")
        plt.title(f'Training and Validation Loss ({suffix})')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        # plt.show()
        plt.savefig(f'{save_to_dir}/ResNet_Loss_{suffix}.png')
        plt.close()

    def save(self, filename: str, file_type: str = ".h5", save_to_dir: str = "./models"):
        """
        Save trained model/data into a file
        :param filename: name of file to save as
        :param file_type: type of file to save as (default: .h5)
        :param save_to_dir: location to save trained model (default: models folder)
        """

        # Validate save_to_dir; if it does not exist, create save_to_dir
        if not os.path.exists(save_to_dir):
            try:  
                os.mkdir(save_to_dir)  
            except OSError as error:  
                print(error)

        # Create path and validate existence; 
        save_to_path = f'{save_to_dir}/{filename}{file_type}'

        # Save training data
        torch.save(
            {
                'epoch': self.epochs,
                'model_state_dict': self.model.state_dict(),
                'model': self.model,
                'optimizer_state_dict': self.optimizer.state_dict(),
                'metrics': self.metrics,
            }, 
	        save_to_path)

