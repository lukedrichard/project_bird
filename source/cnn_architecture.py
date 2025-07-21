import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, pool_kernel=2):
        super(ConvBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            #nn.Dropout(0.3),
            nn.MaxPool2d(kernel_size=pool_kernel)
        )

    def forward(self, x):
        return self.block(x)

#model architecture
class BaseBird(nn.Module):
    def __init__(self, num_classes):
        super(BaseBird, self).__init__()
        self.output_dim = num_classes

        self.model = nn.Sequential(
            ConvBlock(1, 64),
            ConvBlock(64, 128),
            ConvBlock(128, 256),
            ConvBlock(256, 512),
            ConvBlock(512, 1024),

            nn.Conv2d(1024, 2048, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),

            nn.Flatten(),
            nn.Linear(2048, num_classes)
        )

    def forward(self, x):
        return self.model(x)
            

#create and return cnn with num_classes
def get_BaseBird(num_classes):
    return BaseBird(num_classes)

