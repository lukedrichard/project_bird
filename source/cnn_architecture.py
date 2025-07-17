import torch.nn as nn

#model architecture
class BirdCNN(nn.Module):
    
    def __init__(self, num_classes):
        super(BirdCNN, self).__init__()
        
        self.features = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=7, padding=3),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2),

            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2),

            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2),

            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2),

            nn.AdaptiveAvgPool2d((1,1))
        )


        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256,256),
            nn.ReLU(),
            nn.Dropout(0.5),

            #nn.Linear(256,128),
            #nn.ReLU(),
            #nn.Dropout(0.4),

            #nn.Linear(128,64),
            #nn.ReLU(),
            #nn.Dropout(0.4),

            nn.Linear(256, num_classes),
            
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

#create and return cnn with num_classes
def get_BirdCNN(num_classes):
    return BirdCNN(num_classes)

