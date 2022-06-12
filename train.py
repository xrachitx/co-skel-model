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

class DiceLoss(nn.Module):
    def __init__(self, n_classes):
        super(DiceLoss, self).__init__()
        self.n_classes = n_classes

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i  # * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob.unsqueeze(1))
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _dice_loss(self, score, target):
        target = target.float()
        smooth = 1e-5
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        return loss

    def forward(self, inputs, target, weight=None, softmax=False):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        target = self._one_hot_encoder(target)
        if weight is None:
            weight = [1] * self.n_classes
        assert inputs.size() == target.size(), 'predict {} & target {} shape do not match'.format(inputs.size(), target.size())
        class_wise_dice = []
        loss = 0.0
        for i in range(0, self.n_classes):
            dice = self._dice_loss(inputs[:, i], target[:, i])
            class_wise_dice.append(1.0 - dice.item())
            loss += dice * weight[i]
        return loss / self.n_classes


def BCELoss_class_weighted():

    def loss(inpt, target,weights,device):
        inpt = torch.clamp(inpt,min=1e-7,max=1-1e-7)
        inpt = inpt.squeeze()
        target = target.squeeze()
        
        out_im = torch.zeros((target.shape[0],2,448,448))
        out_im[:,0,:,:] = torch.where(target == 0, 1, 0)
        out_im[:,1,:,:] = torch.where(target == 1, 1, 0)
        out_im = out_im.to(device)
#         out_im = out_im.transpose((2, 0, 1))
        
        # print(inpt.shape,target.shape,weights[:,0].shape)
        weights = torch.unsqueeze(weights,axis=2)
        weights = torch.unsqueeze(weights,axis=3)
        weights = torch.tile(weights,(1,1,inpt.shape[-2],inpt.shape[-1]))
        # print(weights[:,0)
#         print(weights[:,0,:,:].shape,out_im.shape,inpt.shape)
        bce = - weights[:,0,:,:] * out_im[:,1,:,:] * torch.log(inpt[:,1,:,:]) - (1 - out_im[:,0,:,:]) * weights[:,1,:,:] * torch.log(1 - inpt[:,0,:,:])
        return torch.mean(bce)

    return loss

def parse_args():
    parser = argparse.ArgumentParser(description='TRAIN CoSkel+')
    parser.add_argument('--batch', default=20, type=int)
    parser.add_argument('--epochs', default=20, type=int)
    parser.add_argument('--checkpoints', default=5, type=int)
    parser.add_argument('--num_classes', default=2, type=int)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--rootDir', default="../../input/co-skel-448x448/CoSkel+", type=str)
    parser.add_argument('--files', default="../../input/co-skel-448x448/CoSkel+/train.csv", type=str)
    parser.add_argument('--device', default="cuda", type=str)
#     parser.add_argument('--freeze_encoder', default=False, type=bool)
#     parser.add_argument('--weighted', default=True, type=bool)
#     parser.add_argument('--weighted', default=True, type=bool)
    parser.add_argument('--freeze_encoder', dest='freeze_encoder', action='store_true',
                    help='Freezing the encoder')
    parser.add_argument('--weighted', dest='weighted', action='store_true',
                    help='using weighted loss')
    parser.add_argument('--class_loss', dest='class_loss', action='store_true',
                    help='using class loss')
    parser.add_argument('--dice_loss', dest='dice_loss', action='store_true',
                    help='using dice loss')

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
    class_loss = args.class_loss
    num_classes = args.num_classes
    dice_loss = args.dice_loss
    print(args.freeze_encoder,args.weighted,args.class_loss,dice_loss)


    try:
        os.makedirs("Checkpoints")
    except:
        print("Checkpoint Folder Exists")

    td = LoadData(files, rootDir,dice_loss)
    train_dataloader = DataLoader(td,batch_size=batch_size,shuffle=True)
    model = Model(device,num_classes,class_loss,freeze_encoder)
    # print(e.parameters())
    for params in model.parameters():
        params.requires_grad = True

    # print(model)
    if weighted:
        criterion = BCELoss_class_weighted()
    
    else:
        criterion = nn.BCELoss()
        
    if class_loss:
        class_criterion = nn.CrossEntropyLoss()
        
    if dice_loss:
        dice_criterion = DiceLoss(num_classes)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(),lr = lr)
    for epoch in tqdm(range(epochs)):
        loss_arr = []
        # print(f"Epoch: {epoch}-------Starting:")
        for i, (img,label,weights,class_label,_) in enumerate(train_dataloader,0):
            img = img.to(device)
            label = label.to(device)
            weights = weights.to(device)
            
            class_label = class_label.to(device)
#             print(img.shape,"class_label: ",class_label.shape, class_label)
            # print("modelling")
            if class_loss:
                pred,class_out = model(img)
#                 print(torch.argmax(class_out,axis=1))
            else:
                pred = model(img)
            # print(torch.unique(pred),torch.unique(label))
            # print("lossing")
#             print("out: ",pred.shape,"label: ",label.shape)
            if weighted and dice_loss:
#                 print("WE HERE")
                loss = criterion(pred,label,weights,device) + dice_criterion(pred,label.squeeze(1),softmax=True)
            elif weighted:
                loss = criterion(pred,label,weights)
            elif dice_loss:
                loss = dice_criterion(pred,label.squeeze(1),softmax=True)
            else:
                loss = criterion(pred,label)
#             print("wloss: ",loss) 
            if class_loss:
                loss += class_criterion(class_out,class_label)
#             print("closs+wloss: ",loss)
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
        
