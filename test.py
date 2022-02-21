import os
from importlib_metadata import files
import numpy as np
import pandas as pd
from torchvision import transforms
import skimage.io as io
import skimage
from torch.utils.data import Dataset, DataLoader
import torch
import cv2
from dataloader import LoadData
# os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

rootDir = "E:\Desktop\College\IP\Datasets\Co-Skel-Plus-Dataset\CoSkel+"
files = "E:\Desktop\College\IP\Datasets\Co-Skel-Plus-Dataset\CoSkel+\train.csv"

td = LoadData(files, rootDir)
train_dataloader = DataLoader(td,)
print(len(train_dataloader))