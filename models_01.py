import torch
from torch import nn
from torchvision import models

class ConvBlock(nn.Module):
  def __init__(self, input_, output_, **kargs):
    super().__init__()
    self.conv = nn.Sequential(
      nn.Conv2d(input_, output_, **kargs),
      nn.BatchNorm2d(output_),
      nn.ReLU()
    )
  def forward(self, X):
    return self.conv(X)


class Inception(nn.Module):
  def __init__(self, input_, ch1x1, ch3x3_red, ch3x3, ch5x5_red, ch5x5, pool_proj):
    super(Inception, self).__init__()
    self.b1 = ConvBlock(input_, ch1x1, kernel_size=1)

    self.b2 = nn.Sequential(
      ConvBlock(input_, ch3x3_red, kernel_size=1),
      ConvBlock(ch3x3_red, ch3x3, kernel_size=3, padding=1)
    )

    self.b3 = nn.Sequential(
      ConvBlock(input_, ch5x5_red, kernel_size=1),
      ConvBlock(ch5x5_red, ch5x5, kernel_size=5, padding=2)
    )
    self.b4 = nn.Sequential(
      nn.MaxPool2d(3, 1, 1),
      ConvBlock(input_, pool_proj, kernel_size=1)
    )
  def forward(self, X):
    b1 = self.b1(X)
    b2 = self.b2(X)
    b3 = self.b3(X)
    b4 = self.b4(X)
    return torch.cat([b1, b2, b3, b4], 1)

class BrainNet_V1(nn.Module):
  def __init__(self):
    super().__init__()
    self.stem = nn.Sequential(
      ConvBlock(2, 24, kernel_size=3, padding=1),
      nn.MaxPool2d(3, 2, 1),
      ConvBlock(24, 32, kernel_size=3, padding=1),
      nn.MaxPool2d(3, 2, 1)
    )
    self.extractor = nn.Sequential(
      Inception(32, 8, 16, 32, 8, 16, 16),
      nn.MaxPool2d(3, 2, 1),
      Inception(72, 16, 32, 64, 16, 32, 32),
      nn.MaxPool2d(3, 2, 1),
    )
    self.clasificator = nn.Sequential(
      nn.AdaptiveAvgPool2d(output_size=(1, 1)),
      nn.Flatten(),
      nn.Linear(144, 3)
    )
  
  def forward(self, X):
    stem = self.stem(X)
    features = self.extractor(stem)
    logits = self.clasificator(features)
    return logits

class BrainNet_V2(nn.Module):
  def __init__(self):
    super().__init__()
    self.stem = nn.Sequential(
      ConvBlock(2, 24, kernel_size=3, padding=1),
      nn.MaxPool2d(3, 2, 1),
      ConvBlock(24, 32, kernel_size=3, padding=1),
      nn.MaxPool2d(3, 2, 1)
    )
    self.extractor = nn.Sequential(
      Inception(32, 8, 16, 32, 8, 16, 16),
      nn.MaxPool2d(3, 2, 1),
      Inception(72, 16, 32, 64, 16, 32, 32),
      nn.MaxPool2d(3, 2, 1),
      Inception(144, 32, 64, 128, 16, 32, 32),
      nn.MaxPool2d(3, 2, 1),
    )
    self.clasificator = nn.Sequential(
      nn.AdaptiveAvgPool2d(output_size=(1, 1)),
      nn.Flatten(),
      nn.Linear(224, 3)
    )
  
  def forward(self, X):
    stem = self.stem(X)
    features = self.extractor(stem)
    logits = self.clasificator(features)
    return logits

class BrainNet_V3(nn.Module):
  def __init__(self):
    super().__init__()
    self.stem = nn.Sequential(
      ConvBlock(2, 24, kernel_size=3, padding=1),
      nn.MaxPool2d(3, 2, 1),
      ConvBlock(24, 32, kernel_size=3, padding=1),
      ConvBlock(32, 32, kernel_size=3, padding=1),
      nn.MaxPool2d(3, 2, 1)
    )
    self.extractor = nn.Sequential(
      Inception(32, 8, 16, 32, 8, 16, 16),
      nn.MaxPool2d(3, 2, 1),
      Inception(72, 16, 32, 64, 16, 32, 32),
      nn.MaxPool2d(3, 2, 1),
      Inception(144, 32, 64, 128, 16, 32, 32),
      nn.MaxPool2d(3, 2, 1),
      Inception(224, 32, 64, 128, 16, 32, 32),
      nn.MaxPool2d(3, 2, 1),
      Inception(224, 32, 64, 128, 16, 32, 32),
    )
    self.clasificator = nn.Sequential(
      nn.AdaptiveAvgPool2d(output_size=(1, 1)),
      nn.Flatten(),
      nn.Linear(224, 64),
      nn.ReLU(),
      nn.Dropout(0.5),
      nn.Linear(64, 3)
    )
  
  def forward(self, X):
    stem = self.stem(X)
    features = self.extractor(stem)
    logits = self.clasificator(features)
    return logits

class BrainNet_V4(nn.Module):
  def __init__(self):
    super().__init__()
    self.stem = nn.Sequential(
      ConvBlock(2, 16, kernel_size=3, padding=1),
      nn.MaxPool2d(3, 2, 1),
      ConvBlock(16, 16, kernel_size=3, padding=1),
      ConvBlock(16, 32, kernel_size=3, padding=1),
      nn.MaxPool2d(3, 2, 1)
    )
    self.extractor = nn.Sequential(
      Inception(32, 8, 16, 32, 8, 16, 16),
      nn.MaxPool2d(3, 2, 1),
      Inception(72, 16, 32, 64, 16, 32, 32),
      nn.MaxPool2d(3, 2, 1),
      Inception(144, 32, 64, 128, 16, 32, 32),
      nn.MaxPool2d(3, 2, 1),
      Inception(224, 32, 64, 128, 16, 32, 32),
      nn.MaxPool2d(3, 2, 1),
      Inception(224, 32, 64, 128, 16, 32, 32),
    )
    self.clasificator = nn.Sequential(
      nn.AvgPool2d(8, 1),
      nn.Flatten(),
      nn.Dropout(0.4),
      nn.Linear(224, 3)
    )
  
  def forward(self, X):
    X = self.stem(X)
    X = self.extractor(X)
    X = self.clasificator(X)
    return X

class BrainMobilenet(nn.Module):
  def __init__(self):
    super().__init__()
    self.backbone = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
    in_features = self.backbone.classifier[1].in_features
    self.backbone.classifier = nn.Identity()
    self.classifier = nn.Sequential(
      nn.Dropout(0.5),
      nn.ReLU(),
      nn.Linear(in_features, 3)
    )
  def forward(self, X):
    backbone = self.backbone(X)
    logits = self.classifier(backbone)
    return logits

class BrainVGG19net(nn.Module):
  def __init__(self):
    super().__init__()
    self.backbone = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1)
    in_features = self.backbone.classifier[0].in_features
    self.backbone.classifier = nn.Identity()
    self.classifier = nn.Sequential(
      nn.Linear(in_features, 4096),
      nn.ReLU(),
      nn.Dropout(0.5),
      nn.Linear(4096, 4096),
      nn.ReLU(),
      nn.Dropout(0.5),
      nn.Linear(4096, 3)
    )
  def forward(self, X):
    backbone = self.backbone(X)
    logits = self.classifier(backbone)
    return logits
