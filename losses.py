import torch.nn as nn


class FBetaLoss(nn.Module):

    def __init__(self, beta=1, epsilon=1e-7):
        super().__init__()
        self.beta = beta
        self.sigmoid = nn.Sigmoid()
        self.epsilon = epsilon

    def forward(self, y_logits, y_true):
        y_pred = self.sigmoid(y_logits)
        TP = (y_pred * y_true).sum(dim=1)
        FP = ((1 - y_pred) * y_true).sum(dim=1)
        FN = (y_pred * (1 - y_true)).sum(dim=1)
        precision = TP / (TP + FP + self.epsilon)
        recall = TP / (TP + FN + self.epsilon)
        fbeta = (1 + self.beta**2) * precision * recall / ((precision * self.beta**2) + recall + self.epsilon)
        fbeta = fbeta.clamp(min=self.epsilon, max=1 - self.epsilon)
        return 1 - fbeta.mean()

    def forward_other(self, y_logits, y_true):
        y_pred = self.sigmoid(y_logits)
        TP = (y_pred * y_true).sum(dim=1)
        FP = ((1 - y_pred) * y_true).sum(dim=1)
        FN = (y_pred * (1 - y_true)).sum(dim=1)
        fbeta = (1 + self.beta**2) * TP / ((1 + self.beta**2) * TP + (self.beta**2) * FN + FP + self.epsilon)
        fbeta = fbeta.clamp(min=self.epsilon, max=1 - self.epsilon)
        return 1 - fbeta.mean()

# TODO: Write some unit tests for these functions
