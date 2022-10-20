from utils import *
import cv2
from torchvision import transforms as T
from config import config

def infer_and_show(path_to_img):
    try:
      backbone = torch.load(config["path_to_save_checkpoint"])
      backbone.to(config['device'])
      print("The trained ckpt is loaded successfully")
    except Exception as e:
      print(e)
      print("There is sth wrong when loading the trained ckpt!")
      
    backbone.eval()
    img = cv2.imread(path_to_img)

    transforms = T.Compose([
            T.ToTensor(),
            T.Resize(config["img_size"])
    ])

    img = transforms(img)
    img = torch.unsqueeze(img, dim=0)
    img = img.to(config["device"])
    landmarks = backbone(img)
    landmarks = landmarks.detach().to("cpu").numpy().tolist()
    
    print(landmarks)

    plot_and_show_landmarks(path_to_img, landmarks)


if __name__ == "__main__":
    from glob import glob
    for img in sorted(glob("/home/rmarefat/Desktop/NSFW_FD/test_images/*")):
      infer_and_show(img)
