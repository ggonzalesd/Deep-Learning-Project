import torch # tensor operations
import time  # timer
from torch import nn # package of layers and activation functions
from tqdm.auto import tqdm # progress bar
#from tqdm import tqdm # bar

class Trainer:
  def __init__(self, model_checkpoint=None, early_stopping=None):
    self.model_checkpoint = model_checkpoint
    self.early_stopping = early_stopping


  def compute_loss_metrics(self, X, y, model, loss_fn, device):
    X = X.to(device)
    y = y.to(device)
    
    logits = model(X)

    loss = loss_fn(logits, y)
    acc = (logits.argmax(1) == y).type(torch.float).sum()

    return loss, acc


  def train_phase(self, train_dl, model, loss_fn, optimizer, device, pbar):
      size = len(train_dl.dataset)

      train_loss, train_acc = 0., 0.

      model.train()

      for batch, (X, y) in enumerate(train_dl): 
          loss, acc = self.compute_loss_metrics(X, y, model, loss_fn, device)

          train_loss += loss.item() * X.shape[0]

          train_acc += acc.item()

          optimizer.zero_grad()
          loss.backward()
          optimizer.step()

          pbar.set_postfix({'loss': loss.item(), 'acc': acc.item()/X.shape[0]})
          pbar.update(1)

      train_loss /= size
      train_acc /= size

      results = {'loss': train_loss, 'acc': train_acc}
      pbar.set_postfix(results)

      return results


  def test_phase(self, test_dl, model, loss_fn, device):
      size = len(test_dl.dataset)

      test_loss, test_acc = 0., 0.

      model.eval()

      pbar = tqdm(total=len(test_dl), desc=f'Validating',  position=0, leave=False)

      with torch.no_grad():
          for batch, (X, y) in enumerate(test_dl):
              loss, acc = self.compute_loss_metrics(X, y, model, loss_fn, device)

              test_loss += loss.item() * X.shape[0]

              test_acc += acc.item()
        
              # Update the progress bar
              pbar.set_postfix({'loss': loss.item(), 'acc': acc.item()/X.shape[0]})
              pbar.update(1)

      test_loss /= size
      test_acc /= size
      
      # Update results per epoch
      results = {'loss': test_loss, 'acc': test_acc}
      pbar.set_postfix(results)
      pbar.close()
      
      return results


  def train(self, train_dl, val_dl, model, num_epochs, optimizer, device='cpu', scheduler=None, leave_bar=False):
      loss_fn = nn.CrossEntropyLoss()

      model = model.to(device)

      train_acc_history, train_loss_history = [], []
      val_acc_history, val_loss_history = [], []  

      for epoch in range(1, num_epochs+1):
          pbar = tqdm(total=len(train_dl), desc=f'Epoch {epoch}/{num_epochs}', leave=leave_bar)

          time_start = time.time()

          train_results = self.train_phase(train_dl, model, loss_fn, optimizer, device, pbar)
          val_results = self.test_phase(val_dl, model, loss_fn, device)

          time_elapsed = time.time() - time_start

          train_loss, train_acc = train_results["loss"], train_results["acc"]
          val_loss, val_acc = val_results["loss"], val_results["acc"]

          results = {'train_loss': train_loss, 'val_loss': val_loss, 
                      'train_acc': train_acc, 'val_acc': val_acc }

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