class EarlyStopping():

  def __init__(self, patience: int, use: bool = True) -> None:
    self.patience = patience
    self.best_loss = None
    self.not_better_count = 0
    self.use = use

  def __call__(self, val_loss: float):
    if self.best_loss is None:
      self.best_loss = val_loss
      return False
    elif val_loss < self.best_loss:
      self.not_better_count = 0
      return False
    else:
      self.not_better_count += 1
      if self.not_better_count >= self.patience:
        return True
      else:
        return False    