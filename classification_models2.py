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


class LowParamCnnWithReLU(nn.Module):
  def __init__(self):
    super().__init__()                                            #   OUTPUTS
    self.net = nn.Sequential(                       # (INITIAL INPUT) 2  512 512
      ConvActDropout(2, 2),                                       #   2  256 256
      ConvActDropout(2, 2),                                       #   2  128 128
      ConvActDropout(2, 32, pool=4),                              #   32  32  32
      ConvActDropout(32, 64, pool=4),                             #   64   8   8
      ConvActDropout(64, 128, pool=4),                            #   128  2   2
    )

    self.classificator = nn.Sequential(
      nn.Flatten(),
      nn.Linear(512, 64),
      nn.ReLU(),
      nn.Linear(64, 3)
    )

  def forward(self, X):
    return self.classificator(self.net(X))


class LowParamCnnWithSELU(nn.Module):
  def __init__(self):
    super().__init__()                                            #   OUTPUTS
    self.net = nn.Sequential(                       # (INITIAL INPUT) 2  512 512
      ConvActDropout(2, 2, activation="SELU"),                    #   2  256 256
      ConvActDropout(2, 2, activation="SELU"),                    #   2  128 128
      ConvActDropout(2, 32, pool=4, activation="SELU"),           #   32  32  32
      ConvActDropout(32, 64, pool=4, activation="SELU"),          #   64   8   8
      ConvActDropout(64, 128, pool=4, activation="SELU"),         #   128  2   2
    )

    self.classificator = nn.Sequential(
      nn.Flatten(),
      nn.Linear(512, 64),
      nn.SELU(),
      nn.Linear(64, 3)
    )

  def forward(self, X):
    return self.classificator(self.net(X))


class HighParamCnnWithReLU(nn.Module):
  def __init__(self):
    super().__init__()                                            #   OUTPUTS
    self.net = nn.Sequential(                       # (INITIAL INPUT) 2  512 512
      ConvActDropout(2, 8),                                       #   8  256 256
      ConvActDropout(8, 32),                                      #   32 128 128
      ConvActDropout(32, 128, pool=4),                            #   128 32  32
      ConvActDropout(128, 256, pool=4),                           #   256  8   8
      ConvActDropout(256, 256),                                   #   256  4   4
      ConvActDropout(256, 512, pool=4),                           #   512  1   1
    )

    self.classificator = nn.Sequential(
      nn.Flatten(),
      nn.Linear(512, 64),
      nn.ReLU(),
      nn.Linear(64, 3)
    )

  def forward(self, X):
    return self.classificator(self.net(X))


class HighParamCnnWithSELU(nn.Module):
  def __init__(self):
    super().__init__()                                            #   OUTPUTS
    self.net = nn.Sequential(                       # (INITIAL INPUT) 2  512 512
      ConvActDropout(2, 8, activation="SELU"),                    #   8  256 256
      ConvActDropout(8, 32, activation="SELU"),                   #   32 128 128
      ConvActDropout(32, 128, pool=4, activation="SELU"),         #   128 32  32
      ConvActDropout(128, 256, pool=4, activation="SELU"),        #   256  8   8
      ConvActDropout(256, 256, activation="SELU"),                #   256  4   4
      ConvActDropout(256, 512, pool=4, activation="SELU"),        #   512  1   1
    )

    self.classificator = nn.Sequential(
      nn.Flatten(),
      nn.Linear(512, 64),
      nn.SELU(),
      nn.Linear(64, 3)
    )

  def forward(self, X):
    return self.classificator(self.net(X))

