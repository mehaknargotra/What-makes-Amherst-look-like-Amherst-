from cam import ClassActivationMap
from resnet_classifier import ResNetClassifier
from process_data import data_loader

# Main function
def main():

    # Directories for binary and non-binary datasets
    binary_dir = "/Users/Preston/CS-682/Final_Project/dataset/Binary_Dataset"
    nonbinary_dir = "/Users/Preston/CS-682/Final_Project/dataset/Nonbinary_Dataset"

    # Boolean switch to use binary or nonbinary data
    use_binary = True

    # Extract and process data from data_loader
    if use_binary:
        extracted_dataloader, dataset_info, val_dataset_imgs = data_loader(dataset_dir=binary_dir)
    else:
        extracted_dataloader, dataset_info, val_dataset_imgs = data_loader(dataset_dir=nonbinary_dir)

    # Get list of validation datasets (Specifically filter for Amherst-only validation)
    val_dataset_imgs = [path for path in val_dataset_imgs if '/Amherst/' in path]
    
    # Set number of images to try out on in validation set
    val_dataset_imgs = val_dataset_imgs[:40]

    # Set number of epochs
    max_epochs = 5

    # Set optimizer name
    optm = "sgd"
    # optm = "adam"

    # Set suffix name for model and graphs
    if use_binary:
        suffix_str = f"bin_{optm}_{16}cit_{max_epochs}ephs"
    else:
        suffix_str = f"nonbin_{optm}_{16}cit_{max_epochs}ephs"

    # Switch on whether to enable training or otherwise
    model_name = f"model_{suffix_str}"
    if True:
        # Train data
        resnet = ResNetClassifier(
            data_loaders=extracted_dataloader,
            num_labels=len(dataset_info.class_lbl),
            optimizer=optm,
            learning_rate=0.01, 
            epochs=max_epochs
        )
        resnet.train()

        # Great and save charts
        resnet.graph(suffix=suffix_str)

        # Save training data in some file (Saves time)
        resnet.save(filename=model_name)
    
    # Run CAM
    cam = ClassActivationMap()
    cam.set_image_paths(val_dataset_imgs)
    cam.extract_model(file_name=model_name)
    cam.run()
    cam.graph(file_suffix=suffix_str)
    print("\n------------------ FINISHED ------------------")


if __name__ == "__main__":
    main()