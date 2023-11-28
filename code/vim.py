import numpy as np
from transformers import ViTModel
from tqdm import tqdm
import torch
import torch.nn as nn
import os
import matplotlib.pyplot as plt

# TODO: Vision Image Transformers (If possible, see if we can implement it; may be difficult)

# Links:
# https://paperswithcode.com/method/vision-transformer
# SOURCE: https://medium.com/mlearning-ai/vision-transformers-from-scratch-pytorch-a-step-by-step-guide-96c3313c2e0c ***


class MyMSA(nn.Module):
    def __init__(self, d, n_heads=2):
        super(MyMSA, self).__init__()
        self.d = d
        self.n_heads = n_heads

        assert d % n_heads == 0, f"Can't divide dimension {d} into {n_heads} heads"

        d_head = int(d / n_heads)
        self.q_mappings = nn.ModuleList([nn.Linear(d_head, d_head) for _ in range(self.n_heads)])
        self.k_mappings = nn.ModuleList([nn.Linear(d_head, d_head) for _ in range(self.n_heads)])
        self.v_mappings = nn.ModuleList([nn.Linear(d_head, d_head) for _ in range(self.n_heads)])
        self.d_head = d_head
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, sequences):
        # Sequences has shape (N, seq_length, token_dim)
        # We go into shape    (N, seq_length, n_heads, token_dim / n_heads)
        # And come back to    (N, seq_length, item_dim)  (through concatenation)
        result = []
        for sequence in sequences:
            seq_result = []
            for head in range(self.n_heads):
                q_mapping = self.q_mappings[head]
                k_mapping = self.k_mappings[head]
                v_mapping = self.v_mappings[head]

                seq = sequence[:, head * self.d_head: (head + 1) * self.d_head]
                q, k, v = q_mapping(seq), k_mapping(seq), v_mapping(seq)

                attention = self.softmax(q @ k.T / (self.d_head ** 0.5))
                seq_result.append(attention @ v)
            result.append(torch.hstack(seq_result))
        return torch.cat([torch.unsqueeze(r, dim=0) for r in result])
    

class MyViTBlock(nn.Module):
    def __init__(self, hidden_d, n_heads, mlp_ratio=4):
        super(MyViTBlock, self).__init__()
        self.hidden_d = hidden_d
        self.n_heads = n_heads

        self.norm1 = nn.LayerNorm(hidden_d)
        self.mhsa = MyMSA(hidden_d, n_heads)

    def forward(self, x):
        out = x + self.mhsa(self.norm1(x))
        return out


class MyViT(nn.Module):
    # SOURCE: https://medium.com/mlearning-ai/vision-transformers-from-scratch-pytorch-a-step-by-step-guide-96c3313c2e0c
    
    def __init__(self, chw, n_patches=7, n_blocks=2, hidden_d=8, n_heads=2, out_d=10):
        # Super constructor
        super(MyViT, self).__init__()
        
        # Attributes
        self.chw = chw # ( C , H , W )
        self.n_patches = n_patches
        self.n_blocks = n_blocks
        self.n_heads = n_heads
        self.hidden_d = hidden_d
        
        # Input and patches sizes
        assert chw[1] % n_patches == 0, "Input shape not entirely divisible by number of patches"
        assert chw[2] % n_patches == 0, "Input shape not entirely divisible by number of patches"
        self.patch_size = (chw[1] / n_patches, chw[2] / n_patches)

        # 1) Linear mapper
        self.input_d = int(chw[0] * self.patch_size[0] * self.patch_size[1])
        self.linear_mapper = nn.Linear(self.input_d, self.hidden_d)
        
        # 2) Learnable classification token
        self.class_token = nn.Parameter(torch.rand(1, self.hidden_d))
        
        # 3) Positional embedding
        self.register_buffer('positional_embeddings', self.get_positional_embeddings(n_patches ** 2 + 1, hidden_d), persistent=False)
        
        # 4) Transformer encoder blocks
        self.blocks = nn.ModuleList([MyViTBlock(hidden_d, n_heads) for _ in range(n_blocks)])
        
        # 5) Classification MLPk
        self.mlp = nn.Sequential(
            nn.Linear(self.hidden_d, out_d),
            nn.Softmax(dim=-1)
        )

    def forward(self, images):
        # Dividing images into patches
        n, c, h, w = images.shape
        patches = self.patchify(images, self.n_patches).to(self.positional_embeddings.device)
        
        # Running linear layer tokenization
        # Map the vector corresponding to each patch to the hidden size dimension
        tokens = self.linear_mapper(patches)
        
        # Adding classification token to the tokens
        tokens = torch.cat((self.class_token.expand(n, 1, -1), tokens), dim=1)
        
        # Adding positional embedding
        out = tokens + self.positional_embeddings.repeat(n, 1, 1)
        
        # Transformer Blocks
        for block in self.blocks:
            out = block(out)
            
        # Getting the classification token only
        out = out[:, 0]
        
        return self.mlp(out) # Map to output dimension, output category distribution
    
    def patchify(images, n_patches):
        n, c, h, w = images.shape

        assert h == w, "Patchify method is implemented for square images only"

        patches = torch.zeros(n, n_patches ** 2, h * w * c // n_patches ** 2)
        patch_size = h // n_patches

        for idx, image in enumerate(images):
            for i in range(n_patches):
                for j in range(n_patches):
                    patch = image[:, i * patch_size: (i + 1) * patch_size, j * patch_size: (j + 1) * patch_size]
                    patches[idx, i * n_patches + j] = patch.flatten()
        return patches

    def get_positional_embeddings(sequence_length, d):
        result = torch.ones(sequence_length, d)
        for i in range(sequence_length):
            for j in range(d):
                result[i][j] = np.sin(i / (10000 ** (j / d))) if j % 2 == 0 else np.cos(i / (10000 ** ((j - 1) / d)))
        return result


# Simple Fully-Connected model example
# NOTE: We will be applied to our model.fc
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


class VIMClassifier():
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
        self.model = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")

        # Add FC Layer
        # self.model.fc = FCModel(input_dim=2048, num_classes=self.num_labels)

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
            param.requires_grad = True

        # for param in self.model.fc.parameters():
        #     param.requires_grad = True

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
        if optimizer == "adam":
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        elif optimizer == "sgd":
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

        # Set number of epochs for training
        self.epochs = epochs

    def train(self) -> None:
        """
        Training code from following link:
        https://dilithjay.com/blog/custom-image-classifier-with-pytorch/
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
                
                # Loop over the batches of data.
                # Each batch in the dataloader is a 2-tuple since our custom dataset has 2 outputs (the image and the label)
                for images, labels in tqdm(self.dataloaders[phase]):
                    
                    # Reset the optimizer’s gradient to zero (else the gradient will get accumulated from previous batches)
                    self.optimizer.zero_grad()

                    # If we’re in the training phase, we keep track of the gradients. 
                    # If not, we don’t keep track of the gradient since it saves compute power and memory. In both cases
                    with torch.set_grad_enabled(phase == 'train'):
                        
                        # We run the images through the model and get the output
                        output = self.model(images.to(self.device))
                        
                        # Turn the labels into one-hot encoded vectors
                        ohe_label = torch.nn.functional.one_hot(labels, num_classes=self.num_labels)

                        # Use the one-hot encoded labels to calculate the loss
                        loss = self.criterion(output, ohe_label.float().to(self.device))

                        # Use argmax to get the predicted labels and use them with the ground truth labels to calculate the accuracies
                        correct_preds = labels.to(self.device) == torch.argmax(output, dim=1)
                        accuracy = (correct_preds).sum()/len(labels)

                    if phase == 'train':
                        # Backpropagate through the loss and calculate the gradients
                        loss.backward()

                        # Update weights as per the calculated gradients
                        self.optimizer.step()

                    # Keep track of the total accuracy, total loss, and batch count, 
                    # since they can be used to calculate the accuracy and loss for the entire epoch.
                    ep_metrics[phase]['loss'] += loss.item()
                    ep_metrics[phase]['accuracy'] += accuracy.item()
                    ep_metrics[phase]['count'] += 1
            
                # Calculate the epoch loss and epoch accuracy, and update the overall metrics dictionary.
                ep_loss = ep_metrics[phase]['loss']/ep_metrics[phase]['count']
                ep_accuracy = ep_metrics[phase]['accuracy']/ep_metrics[phase]['count']

                print(f'Loss: {ep_loss}, Accuracy: {ep_accuracy}\n')

                self.metrics[phase]['loss'].append(ep_loss)
                self.metrics[phase]['accuracy'].append(ep_accuracy)

    def graph(self, suffix: str, save_to_dir: str = "./graphs/resnet_graphs"):
        """
        Show graphs for Loss and Accuracy for both training and validation
        :param suffix: detail of suffix for file name
        """

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
        :param save_to_dir: location of where to save (default: models folder)
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

