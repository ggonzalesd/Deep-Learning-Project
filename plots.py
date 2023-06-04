import torch
from torch.utils import data
import matplotlib.pyplot as plt
from prettytable import PrettyTable

def total_num_parameters(model):
  return sum(p.numel() for p in model.parameters())

def split_dataset(dataset, train_size=0.8):
  n = len(dataset)
  n_train = int(n*train_size)
  n_val = n - n_train
  train_dataset, val_dataset = data.random_split(dataset, [n_train, n_val])
  return train_dataset, val_dataset

def display_random_batch_detector(model, dataloader):
  labels = ['Meningioma', 'Glioma', 'Pituitary']
  X_batch, y_batch = next(iter(dataloader))
  model = model.to('cpu')
  model.eval()
  with torch.no_grad():
    logits = model(X_batch)
    _, preds = torch.max(logits, 1)
    plt.figure(figsize=(14,5))
    for i in range(12):
      image = X_batch[i]*0.5 + 0.5
      image = image[0]*0.5 + image[1]
      true_lab = labels[y_batch[i]]
      pred_lab = labels[preds[i].item()]
      plt.subplot(2, 6, i+1)
      plt.title(f"True: {true_lab}\nPred: {pred_lab}")
      plt.imshow(image.numpy(), cmap='inferno')
      plt.axis('off')
    plt.show()

def plot_model_results(name, results):
  plt.figure(figsize=(14, 5))
  plt.subplot(1, 2, 1)
  plt.title(f"{name} Loss")
  plt.plot(results['train_loss'], label='train loss')
  plt.plot(results['val_loss'], label='val loss')
  plt.legend()
  plt.xlabel("Epochs")
  plt.ylabel("Loss")
  plt.subplot(1, 2, 2)
  plt.title(f"{name} Acc")
  plt.plot(results['train_acc'], label='train acc')
  plt.plot(results['val_acc'], label='val acc')
  plt.xlabel("Epochs")
  plt.ylabel("Accuracy")
  plt.legend()
  plt.show()

def plot_model_compare(results_list):
  plt.figure(figsize=(14, 14))
  plt.subplot(2, 2, 1)
  plt.title("Train Loss")
  plt.xlabel("Epochs")
  plt.ylabel("Loss")
  for name, results in results_list:
    plt.plot(results['train_loss'], label=name+' loss')
  plt.legend()
  plt.subplot(2, 2, 2)
  plt.title("Validation Loss")
  plt.xlabel("Epochs")
  plt.ylabel("Loss")
  for name, results in results_list:
    plt.plot(results['val_loss'], label=name+' loss')
  plt.legend()
  plt.subplot(2, 2, 3)
  plt.title("Train Accuracy")
  plt.xlabel("Epochs")
  plt.ylabel("Accuracy")
  for name, results in results_list:
    plt.plot(results['train_acc'], label=name+' acc')
  plt.legend()
  plt.subplot(2, 2, 4)
  plt.title("Validation Accuracy")
  plt.xlabel("Epochs")
  plt.ylabel("Accuracy")
  for name, results in results_list:
    plt.plot(results['val_acc'], label=name+' acc')
  plt.legend()
  plt.show()

def print_table(headers, values):
    table = PrettyTable(headers)
    for i in values:
        table.add_row(i)
    table.float_format = '.3'
    print(table)

def row(name, ckp_results):
  results = [name]
  if 'train_loss' in ckp_results:
      results += [ckp_results['train_loss'], ckp_results['val_loss']]
  else:
      results += ['-', '-']
  if 'train_acc' in ckp_results:
      results += [ckp_results['train_acc'], ckp_results['val_acc']]
  else:
      results += ['-', '-']
  return results