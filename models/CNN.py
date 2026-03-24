import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights
 
class CNNFeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        ResNet = resnet50(weights=ResNet50_Weights.DEFAULT)
        self.out_channels = 2048

        # Remove the final Global Average Pooling and Fully Connected layers from ResNet
        self.features = nn.Sequential(*list(ResNet.children())[:-2])

    def forward(self, x):
        # input shape : [B, 3, 224, 224]
        x = self.features(x) # output shape : [B, 2048, 7, 7]

        batch_size, channels, height, width = x.size()
        x = x.view(batch_size, channels, height*width) # Shape : [B, 2048, 49]

        x = x.permute(0, 2, 1) # Shape : [B, 49, 2048]

        return x