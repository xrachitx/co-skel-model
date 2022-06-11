import os
import numpy as np
import pandas as pd
from torchvision import transforms
import skimage.io as io
import skimage
from torch.utils.data import Dataset, DataLoader
import torch
import cv2

ANIMOLS = {"Aeroplane": 0, "Bear":1, "Bird":2, "Boat":3, "Bus":4, "Car":5, "Cats":6, "Cow":7, "Cycle":8, "Dog":9, "Elephant": 10, "Giraffe":11, "Horse":12, "Person":13, "Sheep":14, "Zebra":15}

class LoadData(Dataset):
    def __init__(self, fileNames, rootDir, dice_loss,transform=None):
        self.rootDir = rootDir
        self.transform = transform
        self.frame = pd.read_csv(fileNames, dtype=str, delimiter=',', header=None)
    
    def __len__(self):
        return len(self.frame)

    def __getitem__(self, idx):

        inputName = os.path.join(self.rootDir, self.frame.iloc[idx, 0][1:])
        targetName = os.path.join(self.rootDir, self.frame.iloc[idx, 1][1:])
        animol = self.frame.iloc[idx, 1][1:].split("/")[2]
#         print(animol,ANIMOLS[animol])
        # print(inputName,targetName,self.rootDir, self.frame.iloc[idx, 1][1:])
        inputImage = cv2.imread(inputName)
        targetImage = cv2.imread(targetName, cv2.IMREAD_GRAYSCALE)
    
        
        targetImage = targetImage > 0.0
        targetImage = np.expand_dims(targetImage,axis=0)
        if dice_loss:
            out_im = np.zeros((448,448,2))
            out_im[:,:,0] = np.where(targetImage == 0, 1, 0)
            out_im[:,:,1] = np.where(targetImage == 1, 1, 0)
            targetImage = out_im
#             out_im = out_im.astype(np.float32)
#             out_im = out_im.transpose((2, 0, 1))
        counts = np.unique(targetImage,return_counts=True)[1]
        weights = np.array([ counts[0]/(counts[0]+counts[1]) , counts[1]/(counts[0]+counts[1]) ])
        inputImage = inputImage.astype(np.float32)
        targetImage = targetImage.astype(np.float32)
        inputImage = inputImage.transpose((2, 0, 1))
        
#         print("out: ",targetImage.shape)
        
#         return inputImage, targetImage,weights
        return inputImage, targetImage,weights, np.array(ANIMOLS[animol]),self.frame.iloc[idx, 0]

if __name__ == "__main__":
    rootDir ="./CoSkel+"
    files = "./CoSkel+/train.csv"

    td = LoadData(files, rootDir)
    train_dataloader = DataLoader(td,batch_size=20)
    # print(train_dataloader)
    for i, (data) in enumerate(train_dataloader,0):
        print(data[0].shape,data[1].shape,data[2])
        exit()
    # print(len(train_dataloader))
