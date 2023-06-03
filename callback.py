import torch

class ModelCheckpoint:
  def __init__(self, path='checkpoint.pt', mode='min', monitor='val_loss', verbose=False):
      self.path = path
      self.best_score = None
      self.verbose = verbose
      self.mode = mode
      self.monitor = monitor
      self.best_results = None

  def __call__(self, monitor_candidates, model):
      if self.monitor not in monitor_candidates:
          raise ValueError(f"Invalid monitor. Possible values: {monitor_candidates.keys()}") 
      score = monitor_candidates[self.monitor]

      if self.best_score is None or \
          (self.mode == 'min' and score < self.best_score) or \
          (self.mode == 'max' and score > self.best_score):
          if self.verbose:
              if self.best_score != None:
                  print(f"{self.monitor} changed ({self.best_score:.6f} --> {score:.6f}).  Saving model ...\n")
              else:
                  print(f"Saving model...\n")
          self.best_score = score
          self.best_results = monitor_candidates
          self.save_checkpoint(model)

  def save_checkpoint(self, model):
      torch.save(model.state_dict(), self.path)

  def load_checkpoint(self, model):
      model.load_state_dict(torch.load(self.path))
      
      
# Early Stopping
class EarlyStopping:

  def __init__(self, monitor='val_loss', min_delta=0, patience=0, mode="min", verbose=False):
      self.best_score = None
      self.verbose = verbose
      self.mode = mode
      self.monitor = monitor
      self.patience = patience
      self.min_delta = min_delta
      self.counter = 0

  def early_stop(self, monitor_candidates):
      if self.monitor not in monitor_candidates:
          raise ValueError(f"Invalid monitor. Possible values: {monitor_candidates.keys()}") 
      score = monitor_candidates[self.monitor]
      if self.best_score is None:
          self.best_score = score
          if self.verbose:
              print(f"{self.monitor} improved. Best score: {self.best_score:.6f}\n")
      # score has to improve but at least min_delta
      elif(self.mode == 'min' and score > self.best_score - self.min_delta) or \
          (self.mode == 'max' and score < self.best_score + self.min_delta):
          self.counter += 1
          if self.counter >= self.patience:
              return True
      else:
          self.best_score = score
          self.counter = 0
          if self.verbose:
              print(f"{self.monitor} improved. Best score: {self.best_score:.6f}\n")
        
      return False