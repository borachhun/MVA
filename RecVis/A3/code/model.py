import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

nclasses = 250


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv3 = nn.Conv2d(20, 20, kernel_size=5)
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, nclasses)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = F.relu(F.max_pool2d(self.conv3(x), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


class ResNet_50(nn.Module):
    def __init__(self):
        super(ResNet_50, self).__init__()

        # Load pretrained model
        self.model = models.resnet50(weights='IMAGENET1K_V1')

        # Freeze all parameters
        for param in self.model.parameters():
            param.requires_grad = False
        
        # Change classification head
        self.model.fc = nn.Linear(in_features=2048, out_features=nclasses)

        # Unfreeze parameters in classification head
        for param in self.model.fc.parameters():
            param.requires_grad = True

    def forward(self, x):
        return self.model(x)


class ViT_Large(nn.Module):
    def __init__(self):
        super(ViT_Large, self).__init__()

        # Load pretrained model
        self.model = models.vit_l_16(weights='IMAGENET1K_V1')

        # Freeze all parameters
        for param in self.model.parameters():
            param.requires_grad = False
        
        # Change classification head
        self.model.heads = nn.Linear(in_features=1024, out_features=nclasses)

        # Unfreeze parameters in classification head
        for param in self.model.heads.parameters():
            param.requires_grad = True

    def forward(self, x):
        return self.model(x)


class ConvNeXt_Large(nn.Module):
    def __init__(self):
        super(ConvNeXt_Large, self).__init__()

        # Load pretrained model
        self.model = models.convnext_large(weights='IMAGENET1K_V1')

        # Freeze all parameters
        for param in self.model.parameters():
            param.requires_grad = False
        
        # Change classification head
        self.model.classifier[2] = nn.Linear(in_features=1536, out_features=nclasses)

        # Unfreeze parameters in classification head
        for param in self.model.classifier.parameters():
            param.requires_grad = True

    def forward(self, x):
        return self.model(x)
