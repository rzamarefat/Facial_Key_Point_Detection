import torch 
from torch.utils.data import Dataset
from torchvision import transforms as T
from glob import glob
import os
import cv2
from config import config

class Dataset(Dataset):
    def __init__(self, images, root_to_gts, is_train_mode=True):
        self.root_to_gts = root_to_gts

        self.images = images
        self.gts_path = [f for f in sorted(glob(os.path.join(root_to_gts, "*")))]

        self.is_train_mode = is_train_mode


        self.no_aug_transforms = T.Compose([
            T.ToTensor(),
            T.Resize(config["img_size"])
        ])
        self.aug_transforms = T.Compose([
            T. ToTensor(),
            T.Resize(config["img_size"]),
            T.RandomGrayscale(0.3),
            T.RandomHorizontalFlip(0.3),
            T.RandomVerticalFlip(0.3),
            #T.RandomAdjustSharpness(shrapness_factor=3, p=0.5),
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        target_img = self.images[index]
        img = cv2.imread(target_img)

        if self.is_train_mode:
            img = self.aug_transforms(img)
        else:
            img = self.no_aug_transforms(img)

        target_gt = os.path.join(self.root_to_gts, f'{target_img.split("/")[-1].split(".")[0]}.txt')

        with open(target_gt, "r") as h:
            coordinates = h.readlines()

        points = []
        for c in coordinates:
            c = c.replace("\n", "")
            points.append(float(c.split("|")[0]))
            points.append(float(c.split("|")[1]))
        
        img = img.to(config["device"])
        points = torch.tensor(points).to(config["device"])
        return img, points
        
if __name__ == "__main__":
    root_to_images = "/home/rzamarefat/projects/github_projects/facial_keypoint_detector/All_Age_Faces_DS_extracted_MTCNN"
    root_to_gts = "/home/rzamarefat/projects/github_projects/facial_keypoint_detector/All_Age_Faces_DS_extracted_MTCNN_gt"

    ds = Dataset(root_to_images, root_to_gts)

    for c in iter(ds):
        pass
        break