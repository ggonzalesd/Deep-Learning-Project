import torch
from torch import nn
from tqdm import tqdm

"""
def categorical_accuracy(logits, y_true, reduction='sum'):
  # TODO: Obtain predictions from logits
  y_pred = logits.argmax(1)

  # TODO: Compare the predictions with the true value
  acc = (y_pred == y_true).type(torch.float)

  # Return sum or average accuracy 
  if reduction == 'sum':
    return acc.sum()
  if reduction == 'mean':
    return acc.mean()
  else:
    raise ValueError("Invalid 'reduction' argument, only supports 'sum' or 'mean'")
"""

class ModelCheckpoint:
  def __init__(self, path='checkpoint.pt', mode='min', monitor='val_loss', verbose=False):
    self.path = path
    self.best_score = None
    self.mode = mode
    self.monitor = monitor
    self.verbose = verbose
  
  def __call__(self, monitor_canidates, model):
    if self.monitor not in monitor_canidates:
      raise ValueError(f"Invalid monitor. Possible values: {monitor_canidates.keys()}")
    score = monitor_canidates[self.monitor]

    if self.best_score is None or \
      (self.mode == 'min' and score < self.best_score) or \
      (self.mode == 'max' and score > self.best_score):
      if self.verbose:
        if self.best_score != None:
          print(f"{self.monitor} changed ({self.best_score:.6f} -> {score:.6f}). Saving model...\n")
        else:
          print(f"Saving model...\n")
      self.best_score = score
      self.save_checkpoint(model)

  def save_checkpoint(self, model):
    torch.save(model.state_dict(), self.path)
  
  def load_checkpoint(self, model):
    model.load_state_dict(torch.load(self.path))

class Trainer:
  @classmethod
  def train_phase(cls, train_dl, model, loss_fn, optimizer, device):
    n = len(train_dl.dataset)

    train_loss, train_acc = 0., 0.

    for X, y in tqdm(train_dl, total=len(train_dl), desc='Train Phase', ncols=100):
      X = X.to(device)
      y = y.to(device)

      logits = model(X)

      loss = loss_fn(logits, y)

      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

      acc = (logits.argmax(1) == y).type(torch.float).sum().item()
      train_acc += acc

      train_loss += loss.item() * X.shape[0]

    train_loss /= n
    train_acc /= n

    train_dic = {'train_loss': train_loss,
                'train_acc': train_acc}
    return train_dic

  @classmethod
  def validation_phase(cls, val_dl, model, loss_fn, device):
    n = len(val_dl.dataset)
    val_loss, val_acc = 0., 0.
    
    with torch.no_grad():
      for X, y in tqdm(val_dl, total=len(val_dl), desc='Validation Phase', ncols=100):
        X = X.to(device)
        y = y.to(device)

        logits = model(X)

        loss = loss_fn(logits, y)
        acc = (logits.argmax(1) == y).type(torch.float).sum().item()

        val_acc += acc
        val_loss = loss.item() * X.shape[0]

    val_loss /= n
    val_acc /= n

    val_dic = {'val_loss': val_loss,
              'val_acc': val_acc}
    return val_dic

  @classmethod
  def train(cls, train_dl, val_dl, model, optimizer, device, num_epochs, checkpoint=None):
    train_acc_history = []
    val_acc_history = []

    train_loss_history = []
    val_loss_history = []

    loss_fn = nn.CrossEntropyLoss()

    model = model.to(device)

    for epoch in range(num_epochs):
      train_dic = cls.train_phase(train_dl, model, loss_fn, optimizer, device)
      val_dic = cls.validation_phase(val_dl, model, loss_fn, device)

      train_loss, val_loss = train_dic['train_loss'], val_dic['val_loss']
      train_acc, val_acc = train_dic['train_acc'], val_dic['val_acc']
      print(f'\nEpoch ({epoch+1}/{num_epochs}): ' \
            + f'train_loss = {train_loss:>7f}, val_loss= {val_loss:.6f}, ' \
            + f'train_acc = {train_acc:>7f}, val_acc= {val_acc:.6f}\n')

      train_acc_history.append(train_acc)
      val_acc_history.append(val_acc)

      train_loss_history.append(train_loss)
      val_loss_history.append(val_loss)

      if checkpoint != None:
        candidates = {
          'train_loss': train_loss, 'train_acc': train_acc,
          'val_loss': val_loss, 'val_acc': val_acc
        }
        checkpoint(candidates, model)

    model_dic = {'train_acc_history': train_acc_history,
                'val_acc_history': val_acc_history,
                'train_loss_history': train_loss_history,
                'val_loss_history': val_loss_history}

    return model_dic
