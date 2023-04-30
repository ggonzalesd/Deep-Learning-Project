import torch
from torch import nn

def ConvActDropout(input_, output_, kernel=3, pad=1, pool=2, drop=0.2, activation="ReLU"):
  act = { "ReLU": nn.ReLU(), "SELU": nn.SELU() }
  try:
    seq = nn.Sequential(
        nn.Conv2d(input_, output_, kernel_size=kernel, padding=pad),
        act[activation],
        nn.MaxPool2d(pool, stride=pool),
        nn.Dropout(drop)
        )
  except:
    raise ValueError(f"'{activation}' is not allowed")
  return seq


class LowDeepCnnWithReLU(nn.Module):
  def __init__(self):
    super().__init__()                                            #   OUTPUTS
    self.net = nn.Sequential(                       # (INITIAL INPUT) 2  512 512
      ConvActDropout(2, 4, kernel=5, pad=2),                      #   4  256 256
      ConvActDropout(4, 8, pool=4),                               #   8   64  64
      ConvActDropout(8, 32, pool=4),                              #   32  16  16
      ConvActDropout(32, 64, pool=4),                             #   64   4   4
      ConvActDropout(64, 128, pool=4),                            #   128  1   1
    )

    self.classificator = nn.Sequential(
      nn.Flatten(),
      nn.Linear(128, 32),
      nn.ReLU(),
      nn.Dropout(0.5),
      nn.Linear(32, 3)
    )

  def forward(self, X):
    return self.classificator(self.net(X))


class MidDeepCnnWithReLU(nn.Module):
  def __init__(self):
    super().__init__()                                            #   OUTPUTS
    self.net = nn.Sequential(                       # (INITIAL INPUT) 2  512 512
      ConvActDropout(2, 4, kernel=5, pad=2),                      #   4  256 256
      ConvActDropout(4, 8),                                       #   8  128 128
      ConvActDropout(8, 16),                                      #   16  64  64
      ConvActDropout(16, 32, pool=4),                             #   32  16  16
      ConvActDropout(32, 64, pool=4),                             #   64   4   4
      ConvActDropout(64, 128, pool=4),                            #   128  1   1
    )

    self.classificator = nn.Sequential(
      nn.Flatten(),
      nn.Linear(128, 32),
      nn.ReLU(),
      nn.Dropout(0.5),
      nn.Linear(32, 3)
    )

  def forward(self, X):
    return self.classificator(self.net(X))


class HighDeepCnnWithReLU(nn.Module):
  def __init__(self):
    super().__init__()                                            #   OUTPUTS
    self.net = nn.Sequential(                       # (INITIAL INPUT) 2  512 512
      ConvActDropout(2, 4, kernel=5, pad=2),                      #   4  256 256
      ConvActDropout(4, 8),                                       #   8  128 128
      ConvActDropout(8, 16),                                      #   16  64  64
      ConvActDropout(16, 32),                                     #   32  32  32
      ConvActDropout(32, 64),                                     #   64  16  16
      ConvActDropout(64, 128),                                    #   128  8   8
      ConvActDropout(128, 256),                                   #   256  4   4
      ConvActDropout(256, 512, pool=4),                           #   512  1   1
    )

    self.classificator = nn.Sequential(
      nn.Flatten(),
      nn.Linear(512, 128),
      nn.ReLU(),
      nn.Dropout(0.5),
      nn.Linear(128, 32),
      nn.ReLU(),
      nn.Dropout(0.5),
      nn.Linear(32, 3)
    )

  def forward(self, X):
    return self.classificator(self.net(X))

