import torch
import torch.nn
import torch.nn.functional as f
import numpy as np

from skorch import NeuralNetClassifier
from skorch.utils import to_numpy


class LogitNeuralNetClassifier(NeuralNetClassifier):

    def get_loss(self, y_pred, y_true, *args, **kwargs):
        if isinstance(self.criterion_, torch.nn.NLLLoss):
            y_pred = torch.log(y_pred)
        return super().get_loss(y_pred, y_true, *args, **kwargs)

    def predict_proba(self, X):
        y_logits = []
        for yp in self.forward_iter(X, training=False):
            yp = yp[0] if isinstance(yp, tuple) else yp
            y_logits.append(to_numpy(yp))
        y_logits = np.concatenate(y_logits, 0)
        e_y = np.exp(y_logits - np.max(y_logits))
        return e_y / e_y.sum(axis=0)  # only difference
