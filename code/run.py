from resnet_classifier import ResNetClassifier
from process_data import data_loader

# Main function
def main():

    # Directories for binary and non-binary datasets
    binary_dir = "/Users/Preston/CS-682/Final_Project/dataset/Binary_Dataset"
    nonbinary_dir = "/Users/Preston/CS-682/Final_Project/dataset/Nonbinary_Dataset"

    # Extract and process data from data_loader
    extracted_dataloader, dataset_info = data_loader(dataset_dir=binary_dir)

    # Train data
    resnet = ResNetClassifier(
        data_loaders=extracted_dataloader,
        num_labels=len(dataset_info.class_lbl),
        epoches=3
    )
    resnet.train()

    # TODO: Save training data in some file (Saves time)
    resnet.save(filename="model_two_class_v1")


if __name__ == "__main__":
    main()