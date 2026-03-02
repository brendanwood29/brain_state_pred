class EarlyStopping():

  def __init__(self, patience: int) -> None:
    self.pat = patience
    self.best_loss = None
    self.not_better_count = 0

  def check_if_stop(self, val_loss: float) -> bool:

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