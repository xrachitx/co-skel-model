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

def BCELoss_class_weighted():

    def loss(inpt, target,weights):
        inpt = torch.clamp(inpt,min=1e-7,max=1-1e-7)
        inpt = inpt.squeeze()
        target = target.squeeze()
        
        print(inpt.shape,target.shape,weights[:,0].shape)
        weights = torch.unsqueeze(weights,axis=2)
        weights = torch.unsqueeze(weights,axis=3)
        weights = torch.tile(weights,(1,1,inpt.shape[-2],inpt.shape[-1]))
        print(weights[:,0])
        bce = - weights[:,0] * target * torch.log(inpt) - (1 - target) * weights[:,1] * torch.log(1 - inpt)
        return torch.mean(bce)

    return loss

def parse_args():
    parser = argparse.ArgumentParser(description='TRAIN CoSkel+')
    parser.add_argument('--batch', default=20, type=int)
    parser.add_argument('--epochs', default=20, type=int)
    parser.add_argument('--checkpoints', default=5, type=int)
    parser.add_argument('--lr', default=1e-5, type=float)
    parser.add_argument('--rootDir', default="../../input/co-skel-448x448/CoSkel+", type=str)
    parser.add_argument('--files', default="../../input/co-skel-448x448/CoSkel+/train.csv", type=str)
    parser.add_argument('--device', default="cuda", type=str)
    parser.add_argument('--freeze_encoder', default=False, type=bool)
    parser.add_argument('--weighted', default=True, type=bool)

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = parse_args()

    epochs = args.epochs
    rootDir =args.rootDir
    files = args.files
    lr = args.lr
    device = args.device
    freeze_encoder=args.freeze_encoder
    checkpoints = args.checkpoints
    batch_size = args.batch
    weighted = args.weighted


    try:
        os.makedirs("Checkpoints")
    except:
        print("Checkpoint Folder Exists")

    td = LoadData(files, rootDir)
    train_dataloader = DataLoader(td,batch_size=batch_size)
    model = Model(device,freeze_encoder)
    # print(e.parameters())
    for params in model.parameters():
        params.requires_grad = True

    # print(model)
    if weighted:
        criterion = BCELoss_class_weighted()
    else:
        criterion = nn.BCELoss()

    optimizer = torch.optim.Adam(model.parameters(),lr = lr)
    for epoch in tqdm(range(epochs)):
        loss_arr = []
        # print(f"Epoch: {epoch}-------Starting:")
        for i, (img,label,weights) in enumerate(tqdm(train_dataloader,0)):
            img = img.to(device)
            label = label.to(device)
            weights = weights.to(device)
            model = model.to(device)
            # print("modelling")
            pred = model(img)
            # print(torch.unique(pred),torch.unique(label))
            # print("lossing")
            if weighted:
                loss = criterion(pred,label,weights)
            else:
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

        if epoch % checkpoints == 0:
            path = f"./Checkpoints/model_{epoch}.pth"
            torch.save(model.state_dict(), path)

    path = f"./Checkpoints/model_{epochs}.pth"
    torch.save(model.state_dict(), path)
        