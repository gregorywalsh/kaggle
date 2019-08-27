import numpy as np

from skorch import NeuralNetClassifier
from skorch.callbacks import BatchScoring
from skorch.callbacks import Checkpoint
from skorch.callbacks import EpochScoring
from skorch.callbacks import EpochTimer
from skorch.callbacks import PrintLog
from skorch.utils import noop
from skorch.utils import to_numpy
from skorch.utils import train_loss_score
from skorch.utils import valid_loss_score

checkpoint_dir = 'checkpoints'


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
        e_y = np.exp(y_logits - np.max(y_logits))
        return e_y / e_y.sum(axis=0)  # only difference

    def fit(self, X, y=None, **fit_params):
        """See ``NeuralNet.fit``.

        In contrast to ``NeuralNet.fit``, ``y`` is non-optional to
        avoid mistakenly forgetting about ``y``. However, ``y`` can be
        set to ``None`` in case it is derived dynamically from
        ``X``.

        Furthermore, the most recent (best) checkpoint is loaded
        automatically once fitting is complete

        """
        if not self.warm_start or not self.initialized_:
            self.initialize()

        self.partial_fit(X, y, **fit_params)
        self.load_params(
            checkpoint=self._default_callbacks[4][1]  # Default checkpoint callback from list of tuples
        )
        return self

    @property
    def _default_callbacks(self):
        return [
            ('epoch_timer', EpochTimer()),
            ('train_loss', BatchScoring(
                train_loss_score,
                name='train_loss',
                on_train=True,
                target_extractor=noop,
            )),
            ('valid_loss', BatchScoring(
                valid_loss_score,
                name='valid_loss',
                target_extractor=noop,
            )),
            ('valid_acc', EpochScoring(
                'accuracy',
                name='valid_acc',
                lower_is_better=False,
            )),
            ('checkpoint', Checkpoint(
                monitor='valid_loss_best',
                dirname=checkpoint_dir,
            )),
            ('print_log', PrintLog()),
        ]
