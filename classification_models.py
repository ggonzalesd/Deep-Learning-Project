import torch
from torch import nn


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

class FiveLayerSelu(nn.Module):
  def __init__(self):
    super().__init__()
    self.net = nn.Sequential(                                  #   2  512 512
      DownSample(2, 8, batch=False),                           #   8  256 256
      DownSample(8, 16),                                       #   16 128 128
      DownSample(16, 32),                                      #   32  64  64
      DownSample(32, 64),        #   64  16  16
      DownSample(64, 128, activation='SELU'),       #  128   4   4
      DownSample(128, 512, pooling=4, activation='SELU'), 
      DownSample(512, 512, pooling=4, activation='SELU'),     #  512   1   1
    )
    self.classificator = nn.Sequential(
      nn.Flatten(),
      nn.Linear(512, 256),
      nn.SELU(),
      nn.Linear(256, 128),
      nn.SELU(),
      nn.Linear(128, 64),
      nn.SELU(),
      nn.Linear(64, 32),
      nn.SELU(),
      nn.Linear(32, 3),
    )

  def forward(self, X):
    return self.classificator(self.net(X))
  
class FiveLayerLeakyReLU(nn.Module):
  def __init__(self):
    super().__init__()
    self.net = nn.Sequential(                                  #   2  512 512
      DownSample(2, 8, batch=False),                           #   8  256 256
      DownSample(8, 16),                                       #   16 128 128
      DownSample(16, 32),                                      #   32  64  64
      DownSample(32, 64),        #   64  16  16
      DownSample(64, 128, activation='LeakyReLU'),       #  128   4   4
      DownSample(128, 512, pooling=4, activation='LeakyReLU'), 
      DownSample(512, 512, pooling=4, activation='LeakyReLU'),     #  512   1   1
    )
    self.classificator = nn.Sequential(
      nn.Flatten(),
      nn.Linear(512, 256),
      nn.LeakyReLU(0.02),
      nn.Linear(256, 128),
      nn.LeakyReLU(0.02),
      nn.Linear(128, 64),
      nn.LeakyReLU(0.02),
      nn.Linear(64, 32),
      nn.LeakyReLU(0.02),
      nn.Linear(32, 3),
    )

  def forward(self, X):
    return self.classificator(self.net(X))
  
class FiveLayerReLU(nn.Module):
  def __init__(self):
    super().__init__()
    self.net = nn.Sequential(                                  #   2  512 512
      DownSample(2, 8, batch=False),                           #   8  256 256
      DownSample(8, 16),                                       #   16 128 128
      DownSample(16, 32),                                      #   32  64  64
      DownSample(32, 64),        #   64  16  16
      DownSample(64, 128, activation='ReLU'),       #  128   4   4
      DownSample(128, 512, pooling=4, activation='ReLU'), 
      DownSample(512, 512, pooling=4, activation='ReLU'),     #  512   1   1
    )
    self.classificator = nn.Sequential(
      nn.Flatten(),
      nn.Linear(512, 256),
      nn.ReLU(),
      nn.Linear(256, 128),
      nn.ReLU(),
      nn.Linear(128, 64),
      nn.ReLU(),
      nn.Linear(64, 32),
      nn.ReLU(),
      nn.Linear(32, 3),
    )

  def forward(self, X):
    return self.classificator(self.net(X))

