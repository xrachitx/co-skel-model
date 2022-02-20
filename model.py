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
        self.transpose_convs = [nn.ConvTranspose2d(512,512,3),nn.ConvTranspose2d(512,256,3),nn.ConvTranspose2d(512,256,3)]
    
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
        # print(image1.shape)
        # image1
        image1=np.moveaxis(image1,2,0)

        x=Variable(torch.from_numpy(image1)).unsqueeze(0).float()   
        # print(x.shape)

        shape0 = x.shape[-2]
        shape1 = x.shape[-1]
        input_imgs = []
        for i in range(2):
            for j in range(2):
                # print(i*shape0//2,(i+1)*shape0//2,j*shape1//2,(j+1)*shape1//2)
                input_imgs.append(x[:, :,i*shape0//2:(i+1)*shape0//2,j*shape1//2:(j+1)*shape1//2])
                # print("shape: ", input_imgs[-1].shape)
        # x_1 = x[:,:]
        self.vgg_features = []
        for j in range(4):
            img = input_imgs[j]
            vgg = self.vgg16s[j]
            self.vgg_features.append([])
            for i,model in enumerate(vgg):
                img = model(img)
                if i in {4,9,16,23,30}:
                    self.vgg_features[-1].append(img)


        for i in range(len(self.vgg_features[0])):
            for j in range(4):
                print(j,i,self.vgg_features[j][i].shape)
        # enc_out_shape = self.vgg_features[0][-1].shape
        # print(enc_out_shape)
        enc_out =torch.zeros_like(self.vgg_features[0][-2])
        x = self.concat_imgs([self.vgg_features[i][-1] for i in range(4)],enc_out)
        print(x.shape)
        
        for i in range(3,-1,-1):
            x = self.transpose_convs[3-i](x)
            print("x shape: ",x.shape)
            reqd_shape = self.vgg_features[0][i].shape
            # print(reqd_shape)
            # reqd_shape[-1] = reqd_shape[-1]*2
            # reqd_shape[-2] = reqd_shape[-2]*2
            out = torch.zeros((reqd_shape[0], reqd_shape[1], reqd_shape[-2]*2, reqd_shape[-1]*2))
            print(out.shape)
            out = self.concat_imgs([self.vgg_features[j][i] for j in range(4)],out)
        # print(enc_out)
        # for i in range(2):
        #     for j in range(2):
        #         enc_out[]
        # enc_outs = [self.vgg  _features[-1][0],self.vgg_features[-1][1],self.vgg_features[-1][2],self.vgg_features[-1][3]]


        return x

if __name__ == "__main__":
    e = Encoder()
    e("new.png")