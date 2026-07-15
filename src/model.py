import torch
import torch.nn as nn
import torchvision.models as models

class ConvolutionalNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        # Block 1
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=(5, 5), padding=2, stride=1)
        self.bn1 = nn.BatchNorm2d(6)                              # New Layer 1
        self.conv1_2 = nn.Conv2d(6, 6, kernel_size=(3, 3), padding=1)  # New Layer 2
        self.bn1_2 = nn.BatchNorm2d(6)                            # New Layer 3
        self.pool = nn.MaxPool2d(kernel_size=2)
        
        # Block 2
        self.conv2 = nn.Conv2d(6, 16, kernel_size=(5, 5), padding=0, stride=1)  # 112x112 -> 108x108
        self.bn2 = nn.BatchNorm2d(16)                             # New Layer 4
        self.conv2_2 = nn.Conv2d(16, 16, kernel_size=(3, 3), padding=1) # New Layer 5
        self.bn2_2 = nn.BatchNorm2d(16)                           # New Layer 6
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        
        # Block 3
        self.conv3 = nn.Conv2d(16, 32, kernel_size=(3, 3), padding=1, stride=1)
        self.bn3 = nn.BatchNorm2d(32)                             # New Layer 7
        self.conv3_2 = nn.Conv2d(32, 32, kernel_size=(3, 3), padding=1) # New Layer 8
        self.bn3_2 = nn.BatchNorm2d(32)                           # New Layer 9
        self.pool3 = nn.MaxPool2d(kernel_size=2)
        
        # Block 4
        self.conv4 = nn.Conv2d(32, 64, kernel_size=(3, 3), padding=1, stride=1)
        self.bn4 = nn.BatchNorm2d(64)                             # New Layer 10
        self.conv4_2 = nn.Conv2d(64, 64, kernel_size=(3, 3), padding=1) # New Layer 11
        self.bn4_2 = nn.BatchNorm2d(64)                           # New Layer 12
        self.pool4 = nn.MaxPool2d(kernel_size=2)
        
        # Block 5
        self.conv5 = nn.Conv2d(64, 128, kernel_size=(3, 3), padding=1, stride=1)
        self.bn5 = nn.BatchNorm2d(128)                            # New Layer 13
        self.conv5_2 = nn.Conv2d(128, 128, kernel_size=(3, 3), padding=1) # New Layer 14
        self.bn5_2 = nn.BatchNorm2d(128)                          # New Layer 15
        self.pool5 = nn.MaxPool2d(kernel_size=2)
        
        # Block 6
        self.conv6 = nn.Conv2d(128, 256, kernel_size=(3, 3), padding=1, stride=1)
        self.bn6 = nn.BatchNorm2d(256)                            # New Layer 16
        self.conv6_2 = nn.Conv2d(256, 256, kernel_size=(3, 3), padding=1) # New Layer 17
        self.bn6_2 = nn.BatchNorm2d(256)                          # New Layer 18
        self.pool6 = nn.MaxPool2d(kernel_size=2)
        
        # Fully Connected Layers
        self.fc1 = nn.Linear(in_features=256 * 3 * 3, out_features=120)
        self.dropout_fc1 = nn.Dropout(p=0.3)                      # New Layer 19
        self.fc2 = nn.Linear(120, 84)
        self.dropout_fc2 = nn.Dropout(p=0.3)                      # New Layer 20
        self.fc3 = nn.Linear(84, 20)
        
        # activation function
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        # Block 1
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv1_2(x)
        x = self.bn1_2(x)
        x = self.relu(x)
        x = self.pool(x)
        
        # Block 2
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv2_2(x)
        x = self.bn2_2(x)
        x = self.relu(x)
        x = self.pool2(x)
        
        # Block 3
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.conv3_2(x)
        x = self.bn3_2(x)
        x = self.relu(x)
        x = self.pool3(x)
        
        # Block 4
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu(x)
        x = self.conv4_2(x)
        x = self.bn4_2(x)
        x = self.relu(x)
        x = self.pool4(x)
        
        # Block 5
        x = self.conv5(x)
        x = self.bn5(x)
        x = self.relu(x)
        x = self.conv5_2(x)
        x = self.bn5_2(x)
        x = self.relu(x)
        x = self.pool5(x)
        
        # Block 6
        x = self.conv6(x)
        x = self.bn6(x)
        x = self.relu(x)
        x = self.conv6_2(x)
        x = self.bn6_2(x)
        x = self.relu(x)
        x = self.pool6(x)
        
        # Fully Connected Layers
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout_fc1(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout_fc2(x)
        return self.fc3(x)


class TransferLearningNetwork(nn.Module):
    def __init__(self, num_classes=20, pretrained=True, freeze_backbone=False):
        super().__init__()
        # Load a pre-trained ResNet-34 model
        weights = models.ResNet34_Weights.DEFAULT if pretrained else None
        self.backbone = models.resnet34(weights=weights)
        
        # freeze backbone for feature extraction otherwise fine-tune
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        # Replace the final fully connected layer (classifier head)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(p=0.3),  # Add dropout to reduce overfitting
            nn.Linear(in_features, num_classes)
        )

    def forward(self, x):
        return self.backbone(x)
