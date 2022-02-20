import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        # self.input_shape = input_shape

        self.vgg16 = list(models.vgg16(pretrained = True).features)
        print(self.vgg16)
        # print(vgg16_original)
        # print("-------------")
        # print((list(vgg16_original.children()))[0])
        # self.vgg16 = nn.Sequential((*(list(vgg16_original.children())[0])))
        # print("xxxxxxxxxxxxxx")
        # print()
        # # print(self.vgg16.features)
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

if __name__ == "__main__":
    Encoder()