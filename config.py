config = {
    "root_path_to_images": "/home/rmarefat/Desktop/NSFW_FD/data/All_Age_Faces_DS_extracted_MTCNN",
    "root_path_to_gt": "/home/rmarefat/Desktop/NSFW_FD/data/All_Age_Faces_DS_extracted_MTCNN_gt",
    "path_to_save_checkpoint": "./FKD_MobilenetV2.pt",
    "batch_szie": 500,
    "backbone": "mobilenet_v2",
    "num_of_landmarks": 10,
    "epoch": 30,
    "loss_function": "SmoothL1Loss", #"MSELoss"
    "learning_rate": 3e-4,
    "device": "cuda",
    "img_size": (224, 224)

}