import os
import numpy as np
import pandas as pd
from torchvision import transforms
import skimage.io as io
import skimage
from torch.utils.data import Dataset, DataLoader
import torch
import cv2


class LoadData(Dataset):
    def __init__(self, fileNames, rootDir, transform=None):
        self.rootDir = rootDir
        self.transform = transform
        self.frame = pd.read_csv(fileNames, dtype=str, delimiter=',', header=None)
    
    def __len__(self):
        return len(self.frame)

    def __getitem__(self, idx):
        inputName = os.path.join(self.rootDir, self.frame.iloc[idx, 0])
        targetName = os.path.join(self.rootDir, self.frame.iloc[idx, 1])
        # print(inputName,targetName)
        inputImage = cv2.imread(inputName)
        targetImage = cv2.imread(targetName, cv2.IMREAD_GRAYSCALE)
    
        # inputImage = torch.Tensor(inputImage).cuda()
        # targetImage = torch.Tensor(targetImage).cuda()

        # inputImage = torch.Tensor(inputImage)
        # targetImage = torch.Tensor(targetImage)
        # print(inputImage.shape,targetImage.shape)
        inputImage = inputImage.astype(np.float32)
        targetImage = targetImage > 0.0
        targetImage = targetImage.astype(np.float32)
        inputImage = inputImage.transpose((2, 0, 1))
        targetImage = np.expand_dims(targetImage,axis=0)
        return inputImage, targetImage

if __name__ == "__main__":
    rootDir ="./CoSkel+"
    files = "./CoSkel+/train.csv"

    td = LoadData(files, rootDir)
    train_dataloader = DataLoader(td,batch_size=20)
    # print(train_dataloader)
    for i, (data) in enumerate(train_dataloader,0):
        print(data[0].shape,data[1].shape)
        exit()
    # print(len(train_dataloader))