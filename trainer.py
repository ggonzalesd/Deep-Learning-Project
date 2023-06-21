import torch # tensor operations
import time  # timer
from torch import nn # package of layers and activation functions
from tqdm.auto import tqdm # progress bar
import torchmetrics
#from tqdm import tqdm # bar

class Trainer:
  def __init__(self, model_checkpoint=None, early_stopping=None, three_channels=False):
    self.model_checkpoint = model_checkpoint
    self.early_stopping = early_stopping
    self.f1_score = torchmetrics.F1Score(task='multiclass', num_classes=3, average='weighted')
    self.acc_score = torchmetrics.Accuracy('multiclass', num_classes=3, average='weighted')
    self.three_channels = three_channels

  def compute_loss_metrics(self, X, y, model, loss_fn, device):
    if self.three_channels:
      X = torch.concat([X, X.mean(1, keepdim=True)], 1)
    X = X.to(device)
    y = y.to(device)
    
    logits = model(X)

    loss = loss_fn(logits, y)
    
    acc = self.acc_score(logits, y)
    f1 = self.f1_score(logits, y)

    return loss, acc, f1


  def train_phase(self, train_dl, model, loss_fn, optimizer, device, pbar):
      size = len(train_dl.dataset)

      train_loss, train_acc = 0., 0.

      model.train()
      self.f1_score.reset()
      self.acc_score.reset()

      for batch, (X, y) in enumerate(train_dl): 
          loss, acc, f1 = self.compute_loss_metrics(X, y, model, loss_fn, device)

          train_loss += loss.item() * X.shape[0]

          optimizer.zero_grad()
          loss.backward()
          optimizer.step()

          pbar.set_postfix({'loss': loss.item(), 'acc': acc.item(), 'f1': f1.item()})
          pbar.update(1)

      train_loss /= size
      
      train_f1 = self.f1_score.compute().item()
      train_acc = self.acc_score.compute().item()

      results = {'loss': train_loss, 'acc': train_acc, 'f1': train_f1}
      pbar.set_postfix(results)

      return results


  def test_phase(self, test_dl, model, loss_fn, device):
      size = len(test_dl.dataset)

      test_loss, test_acc = 0., 0.

      model.eval()
      self.f1_score.reset()
      self.acc_score.reset()

      pbar = tqdm(total=len(test_dl), desc=f'Validating',  position=0, leave=False)

      with torch.no_grad():
          for batch, (X, y) in enumerate(test_dl):
              loss, acc, f1 = self.compute_loss_metrics(X, y, model, loss_fn, device)

              test_loss += loss.item() * X.shape[0]
        
              # Update the progress bar
              pbar.set_postfix({'loss': loss.item(), 'acc': acc.item(), 'f1': f1.item()})
              pbar.update(1)

      test_loss /= size
      
      test_f1 = self.f1_score.compute().item()
      test_acc = self.acc_score.compute().item()

      # Update results per epoch
      results = {'loss': test_loss, 'acc': test_acc, 'f1': test_f1}
      pbar.set_postfix(results)
      pbar.close()
      
      return results


  def train(self, train_dl, val_dl, model, num_epochs, optimizer, device='cpu', scheduler=None, leave_bar=False):
      loss_fn = nn.CrossEntropyLoss()
      self.f1_score = self.f1_score.to(device)
      self.acc_score = self.acc_score.to(device)

      model = model.to(device)

      train_acc_history, train_loss_history = [], []
      val_acc_history, val_loss_history = [], []  

      for epoch in range(1, num_epochs+1):
          pbar = tqdm(total=len(train_dl), desc=f'Epoch {epoch}/{num_epochs}', position=0, leave=leave_bar)

          time_start = time.time()

          train_results = self.train_phase(train_dl, model, loss_fn, optimizer, device, pbar)
          val_results = self.test_phase(val_dl, model, loss_fn, device)

          time_elapsed = time.time() - time_start

          train_loss, train_acc = train_results["loss"], train_results["acc"]
          val_loss, val_acc = val_results["loss"], val_results["acc"] 
          train_f1, val_f1 = train_results["f1"], val_results["f1"]

          results = {'train_loss': train_loss, 'train_acc': train_acc, 'train_f1': train_f1,
                       'val_loss': val_loss, 'val_acc': val_acc, 'val_f1': val_f1}

          if scheduler is not None:
              scheduler.step()
              lr = scheduler.get_last_lr()[0]
              results['lr'] = lr
          
          if leave_bar:
              pbar.set_postfix(results)
              pbar.close()
          else:
              pbar.close()
              # If no bar, print results
              elapsed = '{:02d}:{:02d}'.format(int(time_elapsed // 60), int(time_elapsed % 60))
              l = []
              for key, value in results.items():
                  l.append(key + "=" + "{:.3f}".format(value))
              results_string = ", ".join(l)

              print(f'Epoch ({epoch}/{num_epochs}): time={elapsed}, ' \
                    + results_string)
              
          # History
          train_loss_history.append(train_loss)
          val_loss_history.append(val_loss)
          train_acc_history.append(train_acc)
          val_acc_history.append(val_acc)

          if self.model_checkpoint is not None:
              self.model_checkpoint(results, model)
          
          # Early stopping
          if self.early_stopping is not None and self.early_stopping.early_stop(results):
              print(f'Early stopped on epoch: {epoch}')
              break

      history = {
          "train_loss": train_loss_history,
          "val_loss": val_loss_history,
          "train_acc": train_acc_history,
          "val_acc": val_acc_history
      }

      return history