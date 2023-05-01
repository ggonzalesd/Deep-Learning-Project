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
    
  class MultiLayerReLU(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            DownSample(2, 8, batch=False),
            DownSample(8, 16),                                  
            DownSample(16, 32),                                 
            DownSample(32, 64),
            DownSample(64, 128),
            DownSample(128, 256),
            DownSample(256, 512),
        )
        self.classificator = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512*4*4, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 10),
        )

    def forward(self, X):
        return self.classificator(self.net(X))


  class MultiLayerSeLU(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            DownSample(2, 8, batch=False),
            DownSample(8, 16),                                  
            DownSample(16, 32),                                 
            DownSample(32, 64),
            DownSample(64, 128),
            DownSample(128, 256),
            DownSample(256, 512),
        )
        self.classificator = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512*4*4, 1024),
            nn.SELU(),
            nn.Linear(1024, 512),
            nn.SELU(),
            nn.Linear(512, 256),
            nn.SELU(),
            nn.Linear(256, 10),
        )

    def forward(self, X):
        return self.classificator(self.net(X))