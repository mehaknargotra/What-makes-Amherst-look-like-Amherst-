from cam import ClassActivationMap
from grad_cam_V2 import GradCAMV2, GradCamModel, gradCamMethod
from test_grad import run_grad_cam
from resnet_classifier import ResNetClassifier
from process_data import data_loader

# Main function
def main():

    # NOTE: Set directories for binary and non-binary datasets from dataset.zip
    bin_data_dir = "/Users/.../dataset/Binary_Dataset"
    nonbin_data_dir = "/Users/.../dataset/Nonbinary_Dataset"

    # Loop through binary and non-binary datasets
    for data_dir in [nonbin_data_dir, bin_data_dir]:

        # Loop through optimizers
        for opt in ["sgd", "adam"]:

            # Loop through maximum number of epochs to train on
            for epoch in [5, 10]:

                # Enable training when running hyperparameter combination for first time
                training_flag = True
                
                # Loop through 
                for cam_name in ["cam", "gradcam", "smoothgradcam"]:
                    
                    # Run code to identify Amherst features
                    identify_amherst(
                        dataset_dir=data_dir,
                        use_binary=(data_dir == bin_data_dir),
                        num_cities=20,
                        num_val_imgs=100,
                        enable_training=training_flag,
                        max_epochs=epoch,
                        optim=opt,
                        learning_rate=0.001,
                        cam_type=cam_name
                    )

                    # Disable training to run same pretrained model for other CAMs
                    training_flag = False


def identify_amherst(dataset_dir: str,
                     use_binary: bool = True,
                     num_cities: int = 19,
                     num_val_imgs: int = 30,
                     enable_training: bool = True,
                     max_epochs: int = 5,
                     optim: str = "adam",
                     learning_rate: float = 0.001,
                     cam_type: str = "smoothgradcam"
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
        "cam" = regular CAM
        "gradcam" = GradCAM
        "smoothgradcam" = Smooth GradCAM
    """
    
    # Extract and process data from data_loader
    extracted_dataloader, dataset_info, val_dataset_imgs = data_loader(dataset_dir=dataset_dir)

    # Get list of validation datasets (Specifically filter for Amherst-only validation)
    val_dataset_imgs = [path for path in val_dataset_imgs if '/Amherst/' in path]
    
    # Set number of images to try out on in validation set
    if num_val_imgs > 0:
        val_dataset_imgs = val_dataset_imgs[:num_val_imgs]

    # Set suffix name for model and graphs
    if use_binary:
        suffix_str = f"bin_{optim}_{num_cities}cit_{max_epochs}ephs_lr{learning_rate}"
    else:
        suffix_str = f"nonbin_{optim}_{num_cities}cit_{max_epochs}ephs_lr{learning_rate}"

    print(f"\nCONFIGURATION: {suffix_str}")
    print(f"CAM TYPE: {cam_type}\n")

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
    cam = ClassActivationMap()
    cam.set_image_paths(val_dataset_imgs)
    cam.extract_model(file_name=model_name)
    if cam_type.lower() == "smoothgradcam":
        cam.run(1)
    elif cam_type.lower() == "gradcam":
        cam.run(2)
    elif cam_type.lower() == "cam":
        cam.run(3)
    else:
        raise Exception("CAM type not supported")
    
    # Graph and save CAM results
    cam.graph(file_suffix=suffix_str+f"_{cam_type}")

    print("\n------------------ FINISHED ------------------")
    

if __name__ == "__main__":
    main()