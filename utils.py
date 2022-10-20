import torch
from torchvision import transforms, models
import numpy as np
import cv2


def plot_and_show_landmarks(image_path, landmarks):
    img = cv2.imread(image_path)
    print(landmarks)    
    img = cv2.circle(img,(int(landmarks[0][0]), int(landmarks[0][1])), 5, (0,255,0), -1)
    img = cv2.circle(img,(int(landmarks[0][2]), int(landmarks[0][3])), 5, (0,255,0), -1)
    img = cv2.circle(img,(int(landmarks[0][4]), int(landmarks[0][5])), 5, (0,255,0), -1)
    img = cv2.circle(img,(int(landmarks[0][6]), int(landmarks[0][7])), 5, (0,255,0), -1)
    img = cv2.circle(img,(int(landmarks[0][7]), int(landmarks[0][8])), 5, (0,255,0), -1)
    
    cv2.imwrite(f"./landmarks_{image_path.split('/')[-1].split('.')[0]}.jpg", img)
    #cv2.imshow("Image", img)
    #cv2.waitKey(0)


def create_backbone(config, use_imagenet_weights=True):
    if config["backbone"] == "mobilenet_v2":
        backbone = models.mobilenet_v2(pretrained=True)
        

        requires_grad = False
        for name, params in backbone.named_parameters():
            if "features.18" in name or "features.17" in name or "classifier" in name:
                requires_grad = True
            params.requires_grad = requires_grad
        
        backbone.classifier[-1] = torch.nn.Linear(backbone.classifier[-1].in_features, config["num_of_landmarks"])
    else:
        raise ValueError("Please provide a valid backbone in name in config!!!")

    return backbone

def train_step(imgs, gts, backbone, criterion, optimizer):
    out = backbone(imgs)
    loss = criterion(out, gts)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()

def validation_step(imgs, gts, backbone, criterion):
    out = backbone(imgs)
    loss = criterion(out, gts)
    loss.backward()

    return loss.item()

if __name__ == "__main__":
    from glob import glob
    import os

    root_path_to_gt = "/home/rzamarefat/projects/github_projects/facial_keypoint_detector/All_Age_Faces_DS_extracted_MTCNN_gt"

    for img_file in sorted(glob("/home/rzamarefat/projects/github_projects/facial_keypoint_detector/All_Age_Faces_DS_extracted_MTCNN/*")):

        path_to_gt = os.path.join(root_path_to_gt, f'{img_file.split("/")[-1].split(".")[0]}.txt')

        plot_and_show_landmarks(img_file, path_to_gt)