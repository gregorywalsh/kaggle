import numpy as np

from scipy.special import softmax
from skorch import NeuralNetClassifier

from skorch.callbacks import EarlyStopping
from skorch.utils import to_numpy

checkpoint_dir = 'checkpoints'


class EarlyStoppingWithLoadBest(EarlyStopping):

    def __init__(self, checkpoint, **kwargs):
        self.checkpoint = checkpoint
        super().__init__(**kwargs)

    def on_train_end(self, net, X=None, y=None, **kwargs):
        net.load_params(checkpoint=self.checkpoint)


class LogitNeuralNetClassifier(NeuralNetClassifier):

    def get_loss(self, y_pred, y_true, *args, **kwargs):
        return super().get_loss(y_pred, y_true, *args, **kwargs)

    def predict_proba(self, X):
        """See ``NeuralNetClassifier.fit``.

        In contrast to ``NeuralNet.fit``, ``y`` is non-optional to
        avoid mistakenly forgetting about ``y``. However, ``y`` can be
        set to ``None`` in case it is derived dynamically from
        ``X``.

        """

        y_logits = []
        for yp in self.forward_iter(X, training=False):
            yp = yp[0] if isinstance(yp, tuple) else yp
            y_logits.append(to_numpy(yp))
        y_logits = np.concatenate(y_logits, 0)
        return softmax(x=y_logits, axis=1)
