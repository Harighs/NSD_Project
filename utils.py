import torch
import torch.nn as nn
import torchvision.models as models
import time

class PerceptualLoss(nn.Module):
    def __init__(self):
        super(PerceptualLoss, self).__init__()
        # Use a pre-trained VGG19 model and extract features from an intermediate layer
        vgg19 = models.vgg19(pretrained=True).features
        self.feature_extractor = nn.Sequential(*list(vgg19.children())[:36]).eval()

        # Freeze the parameters of the feature extractor
        for parameter in self.feature_extractor.parameters():
            parameter.requires_grad = False

    def forward(self, output, target):
        # Extract features for both output and target
        output_features = self.feature_extractor(output)
        target_features = self.feature_extractor(target)

        # Calculate the loss as the mean squared error of the feature representations
        loss = nn.functional.mse_loss(output_features, target_features)
        return loss

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
perceptual_loss = PerceptualLoss().to(device)