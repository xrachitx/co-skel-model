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
import os
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='TRAIN CoSkel+')
    parser.add_argument('--batch', default=20, type=int)
    parser.add_argument('--epochs', default=20, type=int)
    parser.add_argument('--lr', default=1e-5, type=float)

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = parse_args()

    epochs = args.epochs
    rootDir ="../../input/co-skel-448x448/CoSkel+"
    files = "../../input/co-skel-448x448/CoSkel+/train.csv"
    lr = args.lr
    device = "cuda"
    checkpoints = 5
    batch_size = args.batch

    try:
        os.makedirs("Checkpoints")
    except:
        print("Checkpoint Folder Exists")

    td = LoadData(files, rootDir)
    train_dataloader = DataLoader(td,batch_size=batch_size)
    model = Model()
    # print(e.parameters())
    for params in model.parameters():
        params.requires_grad = True

    print(model)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(),lr = lr)
    for epoch in range(tqdm(epochs)):
        loss_arr = []
        # print(f"Epoch: {epoch}-------Starting:")
        for i, (img,label) in enumerate(train_dataloader,0):
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
            path = f"./Checkpoints/model_{epoch}.pth"
            torch.save(model, path)