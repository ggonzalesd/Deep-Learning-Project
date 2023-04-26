import os

import torch
from torch.utils.data import Dataset

import numpy as np
import pandas as pd

class BrainTumorDataset(Dataset):
  def __init__(self, csv_file, root_dir, transform=None):
    self.annotations = pd.read_csv(csv_file)
    self.root_dir = root_dir
    self.transform = transform
  
  def __len__(self):
    return len(self.annotations)

  def __getitem__(self, index):
    path = os.path.join(self.root_dir, self.annotations.iloc[index, 0])
    image = np.load(path)
    y_label = torch.tensor(int(self.annotations.iloc[index, 1]))
    if self.transform:
      image = self.transform(image)
    return (image, y_label)

