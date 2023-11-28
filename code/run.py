from cam import ClassActivationMap
from test_grad import run_grad_cam
from resnet_classifier import ResNetClassifier
from process_data import data_loader

# Main function
def main():

    # Directories for binary and non-binary datasets
    data_dir = "/Users/Preston/CS-682/Final_Project/dataset/Binary_Dataset"
    # data_dir = "/Users/Preston/CS-682/Final_Project/dataset/Nonbinary_Dataset"

    # Code to identify Amherst features
    identify_amherst(
        dataset_dir=data_dir,
        use_binary=False,
        num_cities=19,
        num_val_imgs=30,
        enable_training=True,
        max_epochs=5,
        optim="adam",
        learning_rate=0.001,
        cam_type="reg"
    )

def identify_amherst(dataset_dir: str,
                     use_binary: bool = True,
                     num_cities: int = 19,
                     num_val_imgs: int = 30,
                     enable_training: bool = True,
                     max_epochs: int = 5,
                     optim: str = "adam",
                     learning_rate: float = 0.0001,
                     cam_type: str = "reg"
                     ):
    """
    Main code to run training and CAM
    :param dataset_dir: directory of where to get dataset
    :param use_binary: boolean switch for binary/nonbinary classification
    :param num_cities: number of cities in dataset
    :param num_val_imgs: number of validation images to run on CAM
    :param enable_training: boolean switch to enable training or otherwise
    :param max_epochs: maximum number of epochs to train
    :param optim: optimizer ("sgd" or "adam")
    :param learning_rate: learning rate for training
    :param cam_type: name/type of CAM to run on
        "reg" = regular CAM
        "grad" = Grad-CAM
    """
    
    # Extract and process data from data_loader
    extracted_dataloader, dataset_info, val_dataset_imgs = data_loader(dataset_dir=dataset_dir)

    # Get list of validation datasets (Specifically filter for Amherst-only validation)
    val_dataset_imgs = [path for path in val_dataset_imgs if '/Amherst/' in path]
    
    # Set number of images to try out on in validation set
    val_dataset_imgs = val_dataset_imgs[:num_val_imgs]

    # Set suffix name for model and graphs
    if use_binary:
        suffix_str = f"bin_{optim}_{num_cities}cit_{max_epochs}ephs_lr{learning_rate}_CAM{cam_type}"
    else:
        suffix_str = f"nonbin_{optim}_{num_cities}cit_{max_epochs}ephs_lr{learning_rate}_CAM{cam_type}"

    print(f"CONFIGURATION: {suffix_str}")

    # Switch on whether to enable training or otherwise
    model_name = f"model_{suffix_str}"
    if enable_training:
        # Train data
        resnet = ResNetClassifier(
            data_loaders=extracted_dataloader,
            num_labels=len(dataset_info.class_lbl),
            optimizer=optim,
            learning_rate=learning_rate, 
            epochs=max_epochs
        )
        resnet.train()

        # Create and save charts
        resnet.graph(suffix=suffix_str)

        # Save model
        resnet.save(filename=model_name)
    
    # Run CAM
    if cam_type.lower() == "reg":
        # Regular CAM
        cam = ClassActivationMap()
        cam.set_image_paths(val_dataset_imgs)
        cam.extract_model(file_name=model_name)
        cam.run()
        cam.graph(file_suffix=suffix_str)
    elif cam_type.lower() == "grad":
        # Grad CAM
        cam = ClassActivationMap()
        cam.extract_model(file_name=model_name)
        run_grad_cam(model=cam.model, validation_imgs=val_dataset_imgs)
    else:
        raise Exception("CAM type not supported")

    print("\n------------------ FINISHED ------------------")
    

if __name__ == "__main__":
    main()