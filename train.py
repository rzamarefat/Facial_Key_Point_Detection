import torch
from config import config 
from torch.utils.data import DataLoader
from torchvision import transforms, models
from dataset import Dataset
from glob import glob
import random
import os
from tqdm import tqdm
from utils import *

def run_training_engine():
    backbone = create_backbone(config)

    if config["device"] == "cuda":
        if not(torch.cuda.is_available()):
            raise ValueError("The device in config is cuda but there is no GPU available!")
    
    backbone.to(config["device"])

    data = [f for f in sorted(glob(os.path.join(config["root_path_to_images"], "*")))]
    
    for i in range(10):
        random.shuffle(data)

    val_images = data[0: int(0.1 * len(data))]
    train_images = data[int(0.1 * len(data)):]

    print(f"Number of train samples: {len(train_images)}")
    print(f"Number of val samples: {len(val_images)}")

    train_ds = Dataset(train_images, config["root_path_to_gt"])
    val_ds = Dataset(val_images, config["root_path_to_gt"])

    train_dataloader = DataLoader(train_ds, shuffle=True, batch_size=config["batch_szie"])
    val_dataloader = DataLoader(val_ds, shuffle=True, batch_size=config["batch_szie"])


    if config["loss_function"] == "SmoothL1Loss":
        criterion = torch.nn.SmoothL1Loss()
    elif config["loss_function"] == "MSELoss":
        criterion = torch.nn.MSELoss()
    else:
        raise ValueError("Please provide a valid loss function")

    optimizer = torch.optim.Adam(backbone.parameters(), lr=config["learning_rate"])
    
    
    
    best_val_loss = 100000
    for epoch in range(1, config["epoch"] + 1):
        print(f"Epoch: {epoch}")


        # Train Step
        train_loss_holder = []
        for imgs, gts in tqdm(train_dataloader):
            backbone.train()
            loss = train_step(imgs, gts, backbone, criterion, optimizer)
            train_loss_holder.append(loss)


        validation_loss_holder = []
        for imgs, gts in val_dataloader:
            backbone.eval()
            loss = validation_step(imgs, gts, backbone, criterion)
            validation_loss_holder.append(loss)

        
        avg_train_loss = round(sum(train_loss_holder)/len(train_loss_holder), 3)
        avg_val_loss = round(sum(validation_loss_holder)/len(validation_loss_holder), 3)
        
        
        if avg_val_loss < best_val_loss:
          best_val_loss = avg_val_loss
          torch.save(backbone, config["path_to_save_checkpoint"]) 
        
        report = f"Epoch: {epoch} | Train loss: {avg_train_loss} |  Val loss: {avg_val_loss} "

        print(report)
    

if __name__ == "__main__":
    run_training_engine()