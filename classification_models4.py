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
            nn.Linear(256, 128),
            nn.SELU(),
            nn.Linear(128, 10),
        )

  def forward(self, X):
      return self.classificator(self.net(X))
