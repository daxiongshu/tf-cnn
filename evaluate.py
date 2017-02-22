import pandas as pd
import numpy as np
import sys
import scipy as sp
def multiclass_log_loss(y_true, y_pred, eps=1e-15):
    """Multi class version of Logarithmic Loss metric.
    https://www.kaggle.com/wiki/MultiClassLogLoss
    Parameters
    ----------
    y_true : array, shape = [n_samples]
            true class, intergers in [0, n_classes - 1)
    y_pred : array, shape = [n_samples, n_classes]
    Returns
    -------
    loss : float
    """
    predictions = np.clip(y_pred, eps, 1 - eps)

    # normalize row sums to 1
    predictions /= predictions.sum(axis=1)[:, np.newaxis]

    actual = np.zeros(y_pred.shape)
    n_samples = actual.shape[0]
    actual[np.arange(n_samples), y_true.astype(int)] = 1
    vectsum = np.sum(actual * np.log(predictions))
    loss = -1.0 / n_samples * vectsum
    return loss
def accuracy(y,yp):
    ytmp = np.argmax(yp,axis=1)
    score = y[y==ytmp].shape[0]*1.0/y.shape[0]
    return score
name = sys.argv[1]
s = pd.read_csv(name)
yp = s.drop(['img','real'],axis=1).values
y = s['real'].values
loss = multiclass_log_loss(y,yp)
acc = accuracy(y,yp)
print("Cross Entropy %.2f Accuracy %.2f"%(loss, acc))
