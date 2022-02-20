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
        vgg16_1 = list(models.vgg16(pretrained = True).features)[:-1]
        vgg16_2 = list(models.vgg16(pretrained = True).features)[:-1]
        vgg16_3 = list(models.vgg16(pretrained = True).features)[:-1]
        vgg16_4 = list(models.vgg16(pretrained = True).features)[:-1]
        self.vgg16s = [vgg16_1,vgg16_2,vgg16_3,vgg16_4]


    def forward(self, x):

        image1 = cv2.imread(x)
        print(image1.shape)
        # image1
        image1=np.moveaxis(image1,2,0)

        x=Variable(torch.from_numpy(image1)).float()   
        print(x.shape)

        shape0 = x.shape[-2]
        shape1 = x.shape[-1]
        input_imgs = []
        for i in range(2):
            for j in range(2):
                input_imgs.append(x[:,i*shape0//2:(i+1)*shape0//2,j*shape1//2:(j+1)*shape1//2])
                print(input_imgs[-1].shape)
        # x_1 = x[:,:]
        self.vgg_features = []
        for j in range(4):
            img = input_imgs[j]
            vgg = self.vgg16s[j]
            self.vgg_features.append([])
            for i,model in enumerate(vgg):
                x = model(x)
                if i in {4,9,16,23}:
                    self.vgg_features[-1].append(x)

        print(x.shape)


        return x

if __name__ == "__main__":
    e = Encoder()
    e("new.png")