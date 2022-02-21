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

        inputImage = cv2.imread(inputName)
        targetImage = cv2.imread(targetName, cv2.IMREAD_GRAYSCALE)
    
        # inputImage = torch.Tensor(inputImage).cuda()
        # targetImage = torch.Tensor(targetImage).cuda()

        inputImage = torch.Tensor(inputImage)
        targetImage = torch.Tensor(targetImage)

        return inputImage, targetImage