import torch
import torch.nn as nn
import torch.nn.init as init
import torchvision.models as models
from torchvision.models import vit_b_16

class DensNetWithHead(nn.Module):
    def __init__(self,  hidden_layer_sizes, dropout_rate, num_classes):
        super(DensNetWithHead, self).__init__()

        # Pretrained DenseNet backbone
        self.backbone = models.densenet121(pretrained=True)
        num_features = self.backbone.classifier.in_features

        # Remove the last classification layer of the backbone
        self.backbone.classifier = nn.Identity()

        # Custom head with hidden layers
        layers = []
        input_size = num_features

        for size in hidden_layer_sizes:
            linear_layer = nn.Linear(input_size, size)
            init.kaiming_uniform_(linear_layer.weight, nonlinearity='relu')
            layers.append(linear_layer)
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(size))
            layers.append(nn.Dropout(dropout_rate))
            input_size = size

        # Output layer
        layers.append(nn.Linear(input_size, num_classes))

        # Assemble the custom head
        self.custom_head = nn.Sequential(*layers)

    def forward(self, x):
        # Forward pass through the backbone
        features = self.backbone(x)
  

        # Forward pass through the custom head
        output = self.custom_head(features)

        return output
    


class ViTWithHead(nn.Module):
    def __init__(self, hidden_layer_sizes, dropout_rate, num_classes, pretrained=True):
        super(ViTWithHead, self).__init__()

        # Load the pretrained Vision Transformer backbone
    
        self.backbone = vit_b_16(pretrained=pretrained)

        # Assuming the ViT model ends with a linear layer and we only use the head output
        num_features = self.backbone.heads[0].in_features

        # Remove the last classification head of the backbone (if exists)
        self.backbone.heads = nn.Identity()

        # Custom head with hidden layers
        layers = []
        input_size = num_features
        for size in hidden_layer_sizes:
            linear_layer = nn.Linear(input_size, size)
            torch.nn.init.kaiming_uniform_(linear_layer.weight, nonlinearity='relu')
            layers.append(linear_layer)
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(size))
            layers.append(nn.Dropout(dropout_rate))
            input_size = size

        # Output layer
        layers.append(nn.Linear(input_size, num_classes))

        # Assemble the custom head
        self.custom_head = nn.Sequential(*layers)

    def forward(self, x):
        # Forward pass through the backbone
        features = self.backbone(x)

        # The output from ViT is usually a tuple, we need only the last hidden state
        if isinstance(features, tuple):
            features = features[0]  # Getting the last hidden state

        # Forward pass through the custom head
        output = self.custom_head(features)

        return output
