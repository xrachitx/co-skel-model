import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import numpy as np
import cv2 
from torch.autograd import Variable

class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        vgg16_1 = list(models.vgg16(pretrained = True).features)
        vgg16_2 = list(models.vgg16(pretrained = True).features)
        vgg16_3 = list(models.vgg16(pretrained = True).features)
        vgg16_4 = list(models.vgg16(pretrained = True).features)
        self.vgg16s = [vgg16_1,vgg16_2,vgg16_3,vgg16_4]
        self.transpose_convs = [nn.ConvTranspose2d(512,512,2,2),nn.ConvTranspose2d(512,256,2,2),nn.ConvTranspose2d(256,128,2,2),nn.ConvTranspose2d(128,64,2,2),nn.ConvTranspose2d(64,3,2,2)]
    
    def concat_imgs(self, inps,out):
        # out = 
        x = out.shape[-2]
        y = out.shape[-1]
        cnt = 0
        for i in range(2):
            for j in range(2):
                out[:,:, i*x//2 : (i+1)*x//2 ,  j*x//2 : (j+1)*x//2] += inps[cnt]
                cnt += 1
        return out


    def forward(self, x):

        image1 = cv2.imread(x)
        image1=np.moveaxis(image1,2,0)
        x=Variable(torch.from_numpy(image1)).unsqueeze(0).float()   
        shape0 = x.shape[-2]
        shape1 = x.shape[-1]
        input_imgs = []
        for i in range(2):
            for j in range(2):
                input_imgs.append(x[:, :,i*shape0//2:(i+1)*shape0//2,j*shape1//2:(j+1)*shape1//2])

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
        
        for i in range(3,-1,-1):
            x = self.transpose_convs[3-i](x)
            
            reqd_shape = self.vgg_features[0][i].shape
            out = torch.zeros((reqd_shape[0], reqd_shape[1], reqd_shape[-2]*2, reqd_shape[-1]*2),dtype= self.vgg_features[0][i].dtype)
            out = self.concat_imgs([self.vgg_features[j][i] for j in range(4)],out)
            x = x+out
        x = self.transpose_convs[-1](x)
        return x

if __name__ == "__main__":
    e = Encoder()
    e("new.png")