import torch
import torch.nn.functional as F
from torch import nn


class Linear(nn.Module):

    def __init__(self):
        super().__init__()
        self.layer_1 = nn.Linear(784, 10)

    def forward(self, x):
        x = torch.flatten(x, start_dim=1)
        x = self.layer_1(x)
        return x


class MLP(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.layer_1 = nn.Linear(784, 400)
        self.layer_2 = nn.Linear(400, 10)

    def forward(self, x):
        x = torch.flatten(x, start_dim=1)

        x = self.layer_1(x)
        x = F.relu(x)

        x = self.layer_2(x)

        return x


class MLP_many_layer(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.layer_1 = nn.Linear(784, 200)
        self.layer_2 = nn.Linear(200, 150)
        self.layer_3 = nn.Linear(150, 100)
        self.layer_4 = nn.Linear(100, 10)

    def forward(self, x):
        x = torch.flatten(x, start_dim=1)

        x = self.layer_1(x)
        x = F.relu(x)

        x = self.layer_2(x)
        #x = F.relu(x)

        x = self.layer_3(x)
        #x = F.relu(x)

        x = self.layer_4(x)

        return x


class CNN4Layer(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_1 = nn.Conv2d(1, 8, 3) # size  = 26 #parameter1: antallet af inputs (antallet af billeder), parameter2: amntal output, parameter3: kernelsize NB! out is our own choice, number of channels out in the network
        self.conv_2 = nn.Conv2d(8, 16, 3) #size = 24 maxpool 12
        self.conv_3 = nn.Conv2d(16, 32, 3) #size = 10
        self.conv_4 = nn.Conv2d(32, 32, 3) #size = 8 max pool 4
        self.layer_1 = nn.Linear(16*32, 80) #sizen er nu 4x4 = 16, og der er 16 channels ud fra conv2, derfor 16*16
        self.layer_2 = nn.Linear(80, 10)

    def forward(self, picture):
        # print(picture.mean())
        imageConv1 = F.relu(self.conv_1(picture)) # 1 image is given, 8 comes out, kernel size 9, size of the new images is 28-9+1 = 20
        imageConv2 = F.relu(self.conv_2(imageConv1)) # 8 images is given, 16 comes out, kernel size 3, new image size = 10-3+1 = 8 divides the size of the image with 2
        maxPool2 = F.max_pool2d(imageConv2, 2, 2) #divides the imagesize by 2, image size = 4
        imageConv3 = F.relu(self.conv_3(maxPool2)) # 8 images is given, 16 comes out, kernel size 3, new image size = 10-3+1 = 8 divides the size of the image with 2
        imageConv3 = F.relu(self.conv_4(imageConv3)) # 8 images is given, 16 comes out, kernel size 3, new image size = 10-3+1 = 8 divides the size of the image with 2
        maxPool4 = F.max_pool2d(imageConv3, 2, 2)
        imageFlatten = torch.flatten(maxPool4, start_dim=1)
        linearImage1 = F.relu(self.layer_1(imageFlatten))
        return self.layer_2(linearImage1)


class BasicNetwork(nn.Module):

    def __init__(self):
        super().__init__()
        self.layer_1 = nn.Linear(784, 100)
        self.layer_2 = nn.Linear(100, 10)

    def forward(self, x):
        x = torch.flatten(x, start_dim=1)

        x = self.layer_1(x)
        x = F.relu(x)

        x = self.layer_2(x)

        return x


class CNN(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv_1 = nn.Conv2d(1, 8, kernel_size=9)
        self.conv_3 = nn.Conv2d(8, 16, kernel_size=3)
        self.linear_1 = nn.Linear(4**2 * 16, 60)
        self.linear_2 = nn.Linear(60, 10)

    def forward(self, x):
        x = F.relu(self.conv_1(x))
        x = F.max_pool2d(x, kernel_size=2, stride=2)

        x = F.relu(self.conv_3(x))
        x = F.max_pool2d(x, kernel_size=2, stride=2)

        x = torch.flatten(x, start_dim=1)

        x = self.linear_1(x)
        x = F.relu(x)

        x = self.linear_2(x)

        return x


class CNN_8_32_out(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv_1 = nn.Conv2d(1, 8, kernel_size=9)
        self.conv_3 = nn.Conv2d(8, 32, kernel_size=3)
        self.linear_1 = nn.Linear(4**2 * 32, 60)
        self.linear_2 = nn.Linear(60, 10)

    def forward(self, x):
        x = F.relu(self.conv_1(x))
        x = F.max_pool2d(x, kernel_size=2, stride=2)

        x = F.relu(self.conv_3(x))
        x = F.max_pool2d(x, kernel_size=2, stride=2)

        x = torch.flatten(x, start_dim=1)

        x = self.linear_1(x)
        x = F.relu(x)

        x = self.linear_2(x)

        return x


class CNN_8_128_out(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv_1 = nn.Conv2d(1, 8, kernel_size=9)
        self.conv_3 = nn.Conv2d(8, 128, kernel_size=3)
        self.linear_1 = nn.Linear(4**2 * 128, 60)
        self.linear_2 = nn.Linear(60, 10)

    def forward(self, x):
        x = F.relu(self.conv_1(x))
        x = F.max_pool2d(x, kernel_size=2, stride=2)

        x = F.relu(self.conv_3(x))
        x = F.max_pool2d(x, kernel_size=2, stride=2)

        x = torch.flatten(x, start_dim=1)

        x = self.linear_1(x)
        x = F.relu(x)

        x = self.linear_2(x)

        return x


class CNN_32_128_out(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv_1 = nn.Conv2d(1, 32, kernel_size=9)
        self.conv_3 = nn.Conv2d(32, 128, kernel_size=3)
        self.linear_1 = nn.Linear(4**2 * 128, 60)
        self.linear_2 = nn.Linear(60, 10)

    def forward(self, x):
        x = F.relu(self.conv_1(x))
        x = F.max_pool2d(x, kernel_size=2, stride=2)

        x = F.relu(self.conv_3(x))
        x = F.max_pool2d(x, kernel_size=2, stride=2)

        x = torch.flatten(x, start_dim=1)

        x = self.linear_1(x)
        x = F.relu(x)

        x = self.linear_2(x)

        return x


class CNN_smaller_kernel(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv_1 = nn.Conv2d(1, 8, kernel_size=3)
        self.conv_3 = nn.Conv2d(8, 16, kernel_size=3)
        self.linear_1 = nn.Linear(5 * 5 * 16, 60)
        self.linear_2 = nn.Linear(60, 10)

    def forward(self, x):
        x = F.relu(self.conv_1(x))
        x = F.max_pool2d(x, kernel_size=2, stride=2)

        x = F.relu(self.conv_3(x))
        x = F.max_pool2d(x, kernel_size=2, stride=2)

        x = torch.flatten(x, start_dim=1)

        x = self.linear_1(x)
        x = F.relu(x)

        x = self.linear_2(x)

        return x


class CNN_4layer(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv_1 = nn.Conv2d(1, 8, kernel_size=3)
        self.conv_2 = nn.Conv2d(8, 16, kernel_size=3)
        self.conv_3 = nn.Conv2d(16, 32, kernel_size=3)
        self.conv_4 = nn.Conv2d(32, 64, kernel_size=3)
        self.linear_1 = nn.Linear(4 * 4 * 64, 60)
        self.linear_2 = nn.Linear(60, 10)

    def forward(self, x):
        x = F.relu(self.conv_1(x))
        x = F.relu(self.conv_2(x))
        x = F.max_pool2d(x, kernel_size=2, stride=2)

        x = F.relu(self.conv_3(x))
        x = F.relu(self.conv_4(x))
        x = F.max_pool2d(x, kernel_size=2, stride=2)

        x = torch.flatten(x, start_dim=1)

        x = self.linear_1(x)
        x = F.relu(x)

        x = self.linear_2(x)

        return x


class TopCNN(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv_1 = nn.Conv2d(1, 32, kernel_size=9, padding=4)
        self.bn_1 = nn.BatchNorm2d(32)
        self.conv_2 = nn.Conv2d(32, 32, kernel_size=5, padding=2)
        self.bn_2 = nn.BatchNorm2d(32)

        self.drop_1 = nn.Dropout2d(p=0.2)

        self.conv_3 = nn.Conv2d(32, 64, kernel_size=3)
        self.bn_3 = nn.BatchNorm2d(64)
        self.conv_4 = nn.Conv2d(64, 64, kernel_size=3)
        self.bn_4 = nn.BatchNorm2d(64)

        self.drop_2 = nn.Dropout2d(p=0.2)

        self.linear_1 = nn.Linear(5**2 * 64, 100)

        self.drop_3 = nn.Dropout(p=0.2)

        self.linear_2 = nn.Linear(100, 10)

    def forward(self, x):
        x = F.relu(self.conv_1(x))
        x = self.bn_1(x)
        x = F.relu(self.conv_2(x))
        x = self.bn_2(x)
        x = F.max_pool2d(x, kernel_size=2, stride=2)

        x = self.drop_1(x)

        x = F.relu(self.conv_3(x))
        x = self.bn_3(x)
        x = F.relu(self.conv_4(x))
        x = self.bn_4(x)
        x = F.max_pool2d(x, kernel_size=2, stride=2)

        x = self.drop_2(x)

        x = torch.flatten(x, start_dim=1)

        x = self.linear_1(x)
        x = F.relu(x)

        x = self.drop_3(x)

        x = self.linear_2(x)

        return x
