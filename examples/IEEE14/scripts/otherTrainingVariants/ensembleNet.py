import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

class Splitter:
    def __init__(self, n_splits=5, test_size=0.2):
        self.n_splits = n_splits
        self.test_size = test_size

    def split_data(self, X, y):
        kf = KFold(n_splits=self.n_splits, shuffle=True)
        for train_index, val_index in kf.split(X, y):
            X_train, X_val = X[train_index], X[val_index]
            y_train, y_val = y[train_index], y[val_index]
            yield X_train, X_val, y_train, y_val
            
            
class Trainer:
  def __init__(self, models):
    self.models = models
    self.splitter = Splitter()

  def fit(self, X, y):
    i = 0
    for split in self.splitter.split_data(X, y):
      X_train, X_val, y_train, y_val = split
      self.models[i].fit(X_train, y_train, epochs=500)
      i += 1

  def predict(self, X):
    preds = []
    for m in self.models:
      preds.append(m.predict(X))

    f_pred = np.zeros_like(preds[0])
    for i in range(len(preds)):
      f_pred += preds[i]

    f_pred /= len(preds)

    return f_pred

  def evaluate(self, X_test, y_test):
    return mean_squared_error(y_test, self.predict(X_test))