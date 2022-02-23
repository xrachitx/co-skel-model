from importlib.resources import path
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import numpy as np
import cv2 
from torch.autograd import Variable
from dataloader import LoadData
from torch.utils.data import Dataset, DataLoader
from model import Model
from tqdm import tqdm


epochs = 2
rootDir ="../../input/co-skel-448x448/CoSkel+"
files = "../../input/co-skel-448x448/CoSkel+/train.csv"
lr = 1e-5
device = "cuda"
checkpoints = 1


td = LoadData(files, rootDir)
train_dataloader = DataLoader(td,batch_size=20)
model = Model()
# print(e.parameters())
for params in model.parameters():
    params.requires_grad = True

print(model)
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(),lr = lr)
for epoch in range(epochs):
    loss_arr = []
    for i, (img,label) in enumerate(tqdm(train_dataloader,0)):
        img = img.to(device)
        label = label.to(device)
        model = model.to(device)
        # print("modelling")
        pred = model(img)
        # print(torch.unique(pred),torch.unique(label))
        # print("lossing")
        loss = criterion(pred,label)
        optimizer.zero_grad()
        # print("backing")
        loss.backward()
        optimizer.step()
        loss_arr.append(loss.item())

        # break
    print(f"Epoch: {epoch}-------Loss: {np.mean(loss_arr)}")
    file = open('logs.csv','a+')
    file.write(f"{epoch},{np.mean(loss_arr)}\n")
    file.close()

    if epochs % checkpoints == 0:
        path = f"model_{epoch}.pth"
        torch.save(model, path)