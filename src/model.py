import torch
import torch.nn as nn

class ConvolutionalNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=(5, 5), padding=2, stride=1)
        self.pool = nn.MaxPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=(5, 5), padding=0, stride=1)  # 112x112 -> 108x108
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        
        self.conv3 = nn.Conv2d(16, 32, kernel_size=(3, 3), padding=1, stride=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2)
        self.conv4 = nn.Conv2d(32, 64, kernel_size=(3, 3), padding=1, stride=1)
        self.pool4 = nn.MaxPool2d(kernel_size=2)
        
        self.conv5 = nn.Conv2d(64, 128, kernel_size=(3, 3), padding=1, stride=1)
        self.pool5 = nn.MaxPool2d(kernel_size=2)
        self.conv6 = nn.Conv2d(128, 256, kernel_size=(3, 3), padding=1, stride=1)
        self.pool6 = nn.MaxPool2d(kernel_size=2)
        
        self.fc1 = nn.Linear(in_features=256 * 3 * 3, out_features=120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 20)
        # activation function
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.pool(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.pool2(x))
        
        x = self.relu(self.conv3(x))
        x = self.relu(self.pool3(x))
        x = self.relu(self.conv4(x))
        x = self.relu(self.pool4(x))
        
        x = self.relu(self.conv5(x))
        x = self.relu(self.pool5(x))
        x = self.relu(self.conv6(x))
        x = self.relu(self.pool6(x))
        
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.fc3(x)