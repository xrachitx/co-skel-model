import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import numpy as np
import cv2 
from torch.autograd import Variable
from dataloader import LoadData
from torch.utils.data import Dataset, DataLoader

class Model(nn.Module):
    def __init__(self,device,out_classes,class_pred=False,freeze_encoder=True):
        super().__init__()
        self.class_pred = class_pred
        self.out_classes = out_classes
        vgg1 = models.vgg16(pretrained = True)
        vgg2 = models.vgg16(pretrained = True)
        vgg3 = models.vgg16(pretrained = True)
        vgg4 = models.vgg16(pretrained = True)
        if freeze_encoder:
            for param in vgg1.parameters():
                param.requires_grad = False
                print(param.requires_grad)
            for param in vgg2.parameters():
                param.requires_grad = False
            for param in vgg3.parameters():
                param.requires_grad = False
            for param in vgg4.parameters():
                param.requires_grad = False
        vgg16_1 = nn.ModuleList(list(vgg1.features))
        vgg16_2 = nn.ModuleList(list(vgg2.features))
        vgg16_3 = nn.ModuleList(list(vgg3.features))
        vgg16_4 = nn.ModuleList(list(vgg4.features))

        self.vgg16s = nn.ModuleList([vgg16_1,vgg16_2,vgg16_3,vgg16_4])
        if self.class_pred:
            self.fc_layer = nn.Sequential(
                            nn.Linear(in_features=100352, out_features=100352//4),
                            nn.ReLU(),
                            nn.Linear(in_features=100352//4, out_features=256),
                            nn.ReLU(),
                            nn.Linear(in_features=256, out_features=84),
                            nn.ReLU()
                            )
            self.class_output = nn.Sequential(
                                nn.Linear(in_features=84, out_features=self.out_classes)
                                )
        # print(self.vgg16s)
        # for vgg in self.vgg16s:
            
        self.transpose_convs = nn.ModuleList([nn.Sequential(nn.ConvTranspose2d(512,512,2,2),nn.ReLU()),nn.Sequential(nn.ConvTranspose2d(512,256,2,2),nn.ReLU()),nn.Sequential(nn.ConvTranspose2d(256,128,2,2),nn.ReLU()),nn.Sequential(nn.ConvTranspose2d(128,64,2,2),nn.ReLU()),nn.Sequential(nn.ConvTranspose2d(64,1,2,2))])
        self.sigmoid = nn.Sigmoid()
        self.device = device
    
    def concat_imgs(self, inps,out):
        # out = 
        # inps = inps.to(self.device)
        out = out.to(self.device)
        x = out.shape[-2]
        y = out.shape[-1]
        cnt = 0
        for i in range(2):
            for j in range(2):
                out[:,:, i*x//2 : (i+1)*x//2 ,  j*x//2 : (j+1)*x//2] += inps[cnt]
                cnt += 1
        return out


    def forward(self, x):

        # image1 = cv2.imread(x)
        # image1=np.moveaxis(image1,2,0)
        # x=Variable(torch.from_numpy(image1)).unsqueeze(0).float()   
        shape0 = x.shape[-2]
        shape1 = x.shape[-1]
        input_imgs = []
        for i in range(2):
            for j in range(2):
                
                input_imgs.append(x[:, :,i*shape0//2:(i+1)*shape0//2,j*shape1//2:(j+1)*shape1//2])
                # print(input_imgs[-1].shape)

        self.vgg_features = []
        for j in range(4):
            img = input_imgs[j]
            vgg = self.vgg16s[j]
            self.vgg_features.append([])
            for i,model in enumerate(vgg):
                img = model(img)
                if i in {4,9,16,23,30}:
                    self.vgg_features[-1].append(img)


        enc_out =torch.zeros_like(self.vgg_features[0][-2])
        x = self.concat_imgs([self.vgg_features[i][-1] for i in range(4)],enc_out)
        if self.class_pred:
            y = torch.flatten(x,1)
            y = self.fc_layer(y)
            y = self.class_output(y)
            y = F.softmax(y, dim=1)
#         print(x.shape)
        
        for i in range(3,-1,-1):
            x = self.transpose_convs[3-i](x)
            
            reqd_shape = self.vgg_features[0][i].shape
            out = torch.zeros((reqd_shape[0], reqd_shape[1], reqd_shape[-2]*2, reqd_shape[-1]*2),dtype= self.vgg_features[0][i].dtype)
            out = self.concat_imgs([self.vgg_features[j][i] for j in range(4)],out)
            x = x+out
        
        x = self.transpose_convs[-1](x)
        x = self.sigmoid(x)
        # print(x.shape)
        if not self.class_pred:
            return x
        else:
            return x,y
if __name__ == "__main__":
    rootDir ="./CoSkel+"
    files = "./CoSkel+/train.csv"

#    td = LoadData(files, rootDir)
#    train_dataloader = DataLoader(td,batch_size=20)
    e = Model()
    print(e)
    # print(train_dataloader)
    #for i, (data) in enumerate(train_dataloader,0):
     #   print(data[0].shape,data[1].shape)
      #  e(data[0])

       # exit()
    
    
