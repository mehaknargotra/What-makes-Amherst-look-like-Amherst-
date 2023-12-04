from cam import ClassActivationMap
from grad_cam_V2 import GradCAMV2, GradCamModel, gradCamMethod
from test_grad import run_grad_cam
from resnet_classifier import ResNetClassifier
from process_data import data_loader

# Main function
def main():

    # Directories for binary and non-binary datasets
    # data_dir = "/Users/Preston/CS-682/Final_Project/dataset/Binary_Dataset"
    data_dir = "/Users/Preston/CS-682/Final_Project/dataset/Nonbinary_Dataset"

    # Code to identify Amherst features
    training_flag = False
    for cam_name in ["smoothgradcam", "gradcam", "cam"]:
        identify_amherst(
            dataset_dir=data_dir,
            use_binary=False,
            num_cities=20,
            num_val_imgs=100,
            enable_training=training_flag,
            max_epochs=10,
            optim="adam",
            learning_rate=0.001,
            cam_type=cam_name
        )
        training_flag = False


def identify_amherst(dataset_dir: str,
                     use_binary: bool = True,
                     num_cities: int = 19,
                     num_val_imgs: int = 30,
                     enable_training: bool = True,
                     max_epochs: int = 5,
                     optim: str = "adam",
                     learning_rate: float = 0.0001,
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
        "reg" = regular CAM
        "grad" = Grad-CAM
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
    cam = ClassActivationMap()
    cam.set_image_paths(val_dataset_imgs)
    cam.extract_model(file_name=model_name)
    if cam_type.lower() == "smoothgradcam":
        cam.run(1)
    elif cam_type.lower() == "gradcam":
        cam.run(2)
    elif cam_type.lower() == "cam":
        cam.run(3)
    elif cam_type.lower() == "scorecam":
        cam.run(4)
    elif cam_type.lower() == "sscam":
        cam.run(5)
    elif cam_type.lower() == "iscam":
        cam.run(6)
    # elif cam_type.lower() == "grad_v2":
    #     # Grad CAM (Version 2)
    #     cam = GradCAMV2(name_suffix=suffix_str)
    #     cam.set_image_paths(val_dataset_imgs)
    #     cam.extract_model(file_name=model_name)
    #     # cam.register_hooks()
    #     # cam.process_and_compute_data()
    #     # cam.register_hooks()
    #     gradCamMethod(cam.model, cam.image_paths)
    # elif cam_type.lower() == "grad_v3":
        # cam = ClassActivationMap()
        # cam.extract_model(file_name=model_name)
        # cam.set_image_paths(val_dataset_imgs)
        # device = "cuda" if torch.cuda.is_available() else "cpu"
        # gcmodel = GradCamModel(model=cam.model).to(device)

        # for img_pth in cam.image_paths:
        #     img = imread(img_pth)
        #     img = resize(img, (224,224), preserve_range = True)
        #     img = np.expand_dims(img.transpose((2,0,1)),0)
        #     img /= 255.0
        #     get_mean = np.array([0.485, 0.456, 0.406]).reshape((1,3,1,1))
        #     get_std = np.array([0.229, 0.224, 0.225]).reshape((1,3,1,1))
        #     img = (img - get_mean) / get_std
        #     inpimg = torch.from_numpy(img).to(device, torch.float32)

        #     out, acts = gcmodel(inpimg)
        #     acts = acts.detach().cpu()
        #     if use_binary:
        #         loss = nn.CrossEntropyLoss()(out,torch.from_numpy(np.array([1])).to(device))
        #     else:
        #         loss = nn.BCELoss()(out,torch.from_numpy(np.array([1])).to(device))
        #     loss.backward()

        #     grads = gcmodel.get_act_grads().detach().cpu()
        #     pooled_grads = torch.mean(grads, dim=[0,2,3]).detach().cpu()

        #     for i in range(acts.shape[1]):
        #         acts[:,i,:,:] *= pooled_grads[i]

        #     heatmap_j = torch.mean(acts, dim = 1).squeeze()
        #     heatmap_j_max = heatmap_j.max(axis = 0)[0]
        #     heatmap_j /= heatmap_j_max
            
        #     # Resize heatmap
        #     heatmap_j = resize(heatmap_j, (224,224), preserve_range=True)

        #     # Color mapping
        #     cmap = mpl.cm.get_cmap('jet',256)
        #     heatmap_j2 = cmap(heatmap_j,alpha = 0.2)

        #     fig, axs = plt.subplots(1,1,figsize = (5,5))
        #     axs.imshow((img*get_std+get_mean)[0].transpose(1,2,0))
        #     axs.imshow(heatmap_j2)
        #     plt.show()
    else:
        raise Exception("CAM type not supported")
    
    cam.graph(file_suffix=suffix_str+f"_{cam_type}")

    print("\n------------------ FINISHED ------------------")
    

if __name__ == "__main__":
    main()