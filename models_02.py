import torch
from torch import nn
from torchvision import models

class ConvBlock(nn.Module):
  def __init__(self, input_, output_, **kargs):
    super().__init__()
    self.conv = nn.Sequential(
      nn.Conv2d(input_, output_, **kargs),
      nn.BatchNorm2d(output_),
      nn.SELU()
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

class BrainNetv1(nn.Module):
    def __init__(self):
      super().__init__()
      self.stem = nn.Sequential(
        ConvBlock(2, 12, kernel_size=3, padding=1),
        nn.MaxPool2d(3, 2, 1),
        ConvBlock(12, 24, kernel_size=3, padding=1),
        nn.MaxPool2d(3, 2, 1)
      )
      self.extractor = nn.Sequential(
        Inception(24, 8, 16, 32, 8, 16, 16),
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
        nn.SELU(),
        nn.Dropout(0.5),
        nn.Linear(64, 3)
     )

    def forward(self, X):
      stem = self.stem(X)
      features = self.extractor(stem)
      logits = self.clasificator(features)
      return logits
    

class BrainNetv2(nn.Module):
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
    
class BrainNetv3(nn.Module):
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
      nn.SELU(),
      nn.Dropout(0.3),
      nn.Linear(64, 3)
    )
  
  def forward(self, X):
    stem = self.stem(X)
    features = self.extractor(stem)
    logits = self.clasificator(features)
    return logits

class BrainNetv4(nn.Module):
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
      nn.Dropout(0.5),
      nn.Linear(224, 3)
    )
  
  def forward(self, X):
    X = self.stem(X)
    X = self.extractor(X)
    X = self.clasificator(X)
    return X


class BrainGoogleNet(nn.Module):
    def __init__(self):
      super().__init__()
      self.backbone = models.googlenet(weights='IMAGENET1K_V1')
      in_features = self.backbone.fc.in_features
      self.backbone.fc = nn.Identity()
      self.classifier = nn.Linear(in_features, 3)
    def forward(self, X):
      backbone = self.backbone(X)
      logits = self.classifier(backbone)
      return logits
