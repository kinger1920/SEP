import torch
import torch.nn as nn

class ConvolutionalNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=(5, 5), padding=2,stride=1)
        self.pool = nn.MaxPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=(5, 5), padding=0, stride=1)  # 112x112 -> 108x108
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        self.fc1 = nn.Linear(in_features=16 * 54 * 54, out_features=120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 20)
        # activation function
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        x1 = self.relu(self.conv1(x))
        x2 = self.relu(self.pool(x1))
        x3 = self.relu(self.conv2(x2))
        x4 = self.relu(self.pool2(x3))
        x4 = x4.view(x4.size(0), -1)
        x5 = self.relu(self.fc1(x4))
        x6 = self.relu(self.fc2(x5))
        return self.fc3(x6)