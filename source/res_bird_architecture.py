import torch
import torch.nn as nn
import torch.nn.functional as F


import torch
import torch.nn as nn
import torch.nn.functional as F

class DownsampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DownsampleBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.conv3 = nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, bias=False)

        self.relu = nn.ReLU()

        # Shortcut path
        self.shortcut_pool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.shortcut_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False)
        self.shortcut_bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        # Shortcut path
        identity = self.shortcut_pool(x)
        identity = self.shortcut_conv(identity)
        identity = self.shortcut_bn(identity)

        # Main path
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += identity
        return self.relu(out)



class WideBasicBlock(nn.Module):
    def __init__(self, channels):
        super(WideBasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3,
                               padding=1, stride=1, bias=False)
        self.bn1 = nn.BatchNorm2d(in_channels)

        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3,
                               padding=1, stride=1, bias=False)
        self.bn2 = nn.BatchNorm2d(in_channels)

        self.relu = nn.ReLU()
        # Projection shortcut if needed
        self.shortcut = nn.Sequential()

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += self.shortcut(x)

        return  self.relu(out)


#model architecture
class ResBird(nn.Module):
    def __init__(self, num_classes):
        super(ResBird, self).__init__()
        self.output_dim = num_classes

        self.model = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, stride=1, padding=2),
            nn.MaxPool2d(kernel_size=2, stride=2)

            DownsampleBlock(32, 64),
            WideBasicBlock(64),
            WideBasicBlock(64),

            DownsampleBlock(64, 128),
            WideBasicBlock(128),
            WideBasicBlock(128),

            DownsampleBlock(128, 256),
            WideBasicBlock(256),
            WideBasicBlock(256),

            DownsampleBlock(256, 512),
            WideBasicBlock(512),
            WideBasicBlock(512),

            nn.Conv2d(512, 512, kernel=(4,6), stride=(1,2)),
            nn.BatchNorm2d(),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Conv2d(512, self.output_dim, kernel=1, stride=1),
            nn.BatchNorm2d(),
            nn.Dropout(0.3),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )

    def forward(self, x):
        return self.model(x)
            

#create and return cnn with num_classes
def get_ResBird(num_classes):
    return ResBird(num_classes)