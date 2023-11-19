from cam import ClassActivationMap
from resnet_classifier import ResNetClassifier
from process_data import data_loader

# Main function
def main():

    # Directories for binary and non-binary datasets
    binary_dir = "/Users/Preston/CS-682/Final_Project/dataset/Binary_Dataset"
    nonbinary_dir = "/Users/Preston/CS-682/Final_Project/dataset/Nonbinary_Dataset"

    # Extract and process data from data_loader
    extracted_dataloader, dataset_info, val_dataset_imgs = data_loader(dataset_dir=binary_dir)

    # Get list of validation datasets (Specifically filter for Amherst-only validation)
    val_dataset_imgs = [path for path in val_dataset_imgs if '/Amherst/' in path]
    # Set number of images to try out on in validation set
    val_dataset_imgs = val_dataset_imgs[:40]

    # Switch on whether to enable training or otherwise
    max_epochs = 20
    model_name = f"model_binary_16cities_{max_epochs}epochs_nograd"
    if True:
        # Train data
        resnet = ResNetClassifier(
            data_loaders=extracted_dataloader,
            num_labels=len(dataset_info.class_lbl),
            optimizer="adam",
            learning_rate=0.01, 
            epochs=max_epochs
        )
        resnet.train()

        # Great and save charts
        resnet.graph()

        # Save training data in some file (Saves time)
        resnet.save(filename=model_name)
    
    # Run CAM
    # sample_imgs = "/Users/Preston/CS-682/Final_Project/dataset/Binary_Dataset/Amherst/CS682-1373-Pt9"
    cam = ClassActivationMap()
    cam.set_image_paths(val_dataset_imgs)
    # cam.set_image_batch(val_data, [0, 1])
    # cam.extract_imgs(dir=sample_imgs, num_imgs=50)
    cam.extract_model(file_name=model_name)
    cam.run()
    cam.graph()
    print("\n------------------ FINISHED ------------------")


if __name__ == "__main__":
    main()