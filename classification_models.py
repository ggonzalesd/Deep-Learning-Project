import torch
from torch import nn

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

class InceptionModel(nn.Module):
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

def DownSample(input_, output_, kernel=3, padding=1, pooling=2, batch=True, activation='ReLU'):
  act = None
  if activation == 'ReLU':
    act = nn.ReLU()
  elif activation == 'LeakyReLU':
    act = nn.LeakyReLU(0.02)
  elif activation == 'SELU':
    act = nn.SELU()
  else:
    raise ValueError(f"'{activation}' is not allowed")

  seq = nn.Sequential(nn.Conv2d(input_, output_, kernel_size=kernel, padding=padding))
  if batch:
    seq.append(nn.BatchNorm2d(output_))
  seq.append(act)
  seq.append(nn.MaxPool2d(pooling, stride=pooling))
  return seq

class SimpleModelWithReLU(nn.Module):
  def __init__(self):
    super().__init__()
    self.net = nn.Sequential(                                  #   2  512 512
      DownSample(2, 8, batch=False),                           #   8  256 256
      DownSample(8, 16),                                       #   16 128 128
      DownSample(16, 32),                                      #   32  64  64
      DownSample(32, 64, pooling=4),                           #   64  16  16
      DownSample(64, 128, pooling=4),                          #  128   4   4
      DownSample(128, 512, pooling=4),                         #  512   1   1
    )
    self.classificator = nn.Sequential(
      nn.Flatten(),
      nn.Linear(512, 64),
      nn.ReLU(),
      nn.Linear(64, 3)
    )

  def forward(self, X):
    return self.classificator(self.net(X))

class SimpleModelWithSELU(nn.Module):
  def __init__(self):
    super().__init__()
    self.net = nn.Sequential(                                  #   2  512 512
      DownSample(2, 8, batch=False),                           #   8  256 256
      DownSample(8, 16),                                       #   16 128 128
      DownSample(16, 32),                                      #   32  64  64
      DownSample(32, 64, pooling=4, activation='SELU'),        #   64  16  16
      DownSample(64, 128, pooling=4, activation='SELU'),       #  128   4   4
      DownSample(128, 512, pooling=4, activation='SELU'),      #  512   1   1
    )
    self.classificator = nn.Sequential(
      nn.Flatten(),
      nn.Linear(512, 64),
      nn.SELU(),
      nn.Linear(64, 3)
    )

  def forward(self, X):
    return self.classificator(self.net(X))

class DeeperModelWithLeakyReLU(nn.Module):
  def __init__(self):
    super().__init__()
    self.net = nn.Sequential(                                  #   2  512 512
      DownSample(2, 8, batch=False),                           #   8  256 256
      DownSample(8, 16),                                       #   16 128 128
      DownSample(16, 32),                                      #   32  64  64
      DownSample(32, 64),                                      #   64  32  32
      DownSample(64, 128, activation='LeakyReLU'),             #  128  16  16
      DownSample(128, 512, pooling=4, activation='LeakyReLU'), #  512   4   4
      DownSample(512, 512, pooling=4, activation='LeakyReLU'), #  512   1   1
    )
    self.classificator = nn.Sequential(
      nn.Flatten(),
      nn.Linear(512, 64),
      nn.LeakyReLU(0.02),
      nn.Linear(64, 16),
      nn.LeakyReLU(0.02),
      nn.Linear(16, 3)
    )

  def forward(self, X):
    return self.classificator(self.net(X))

class DeeperModelWithSELU(nn.Module):
  def __init__(self):
    super().__init__()
    self.net = nn.Sequential(                             #   2  512 512
      DownSample(2, 8, batch=False),                      #   8  256 256
      DownSample(8, 16),                                  #   16 128 128
      DownSample(16, 32),                                 #   32  64  64
      DownSample(32, 64),                                 #   64  32  32
      DownSample(64, 128, activation='SELU'),             #  128  16  16
      DownSample(128, 512, pooling=4, activation='SELU'), #  512   4   4
      DownSample(512, 512, pooling=4, activation='SELU'), #  512   1   1
    )
    self.classificator = nn.Sequential(
      nn.Flatten(),
      nn.Linear(512, 64),
      nn.ReLU(),
      nn.Linear(64, 16),
      nn.SELU(),
      nn.Linear(16, 3)
    )

  def forward(self, X):
    return self.classificator(self.net(X))
