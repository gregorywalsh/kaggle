import hypothesis
import nets
import skorch
import skorchextend
import torch
import torch.nn as nn

from distributions import loguniform, randomnewinstance
from functools import partial
from skorch.dataset import CVSplit as Split
from scipy.stats import uniform, binom, randint

checkpoint = skorch.callbacks.Checkpoint(
    monitor='valid_loss_best',
    dirname='challenges/spooky_author_identification/checkpoint'
)

estimator_kwargs = {
    'train_split': Split(cv=0.2, stratified=True),
    'max_epochs': 100,
    'optimizer': torch.optim.Adam,
    'callbacks': [
        ('early_stopping', skorchextend.EarlyStoppingWithLoadBest(
            checkpoint=checkpoint,
            monitor='valid_loss',
            patience=2,
            threshold=0.001,
            threshold_mode='rel',
            lower_is_better=True,
        )),
        ('checkpoint', checkpoint),
    ]
}

rnn_module_kwargs = {
    'freeze_embedding': False,
    'recurrent_depth': 1,
    'recurrent_directions': 1,
    'recurrent_features': 128,
    'fc_hidden_features': 128,
    'fc_hidden_depth': 2,
    'out_features': 3,
    'dropout': 0.5
}

mean_embedding_module_kwargs = {
    'freeze_embedding': False,
    'fc_hidden_features': 128,
    'fc_hidden_depth': 0,
    'out_features': 3,
    'dropout': 0.875
}

optimizer_kwargs = {
    "lr": 0.001,
    # "weight_decay": 0.01
}

iter_train_kwargs = {
    'batch_size': 128,
    'shuffle': True,
    'drop_last': True,
}

iter_valid_kwargs = {
    'batch_size': -1,
    'shuffle': False,
    'drop_last': False,
}


class RNNHypothesis(hypothesis.AbstractNNHypothesis):

    hyper_dists = {

    }

    def __init__(self, hyper_search_strat, hyper_search_kwargs, embeddings, device):

        self.device = device
        self.embeddings = embeddings
        super().__init__(
            mode='classification',
            hyper_search_strat=hyper_search_strat,
            hyper_search_kwargs=hyper_search_kwargs,
            estimator_kwargs={
                'module': nets.RNN,
                'module__embeddings': embeddings,
                'device': self.device,
                'iterator_train__collate_fn': self.generate_batch,
                'iterator_valid__collate_fn': self.generate_batch,
                **estimator_kwargs,
                **hypothesis.prefix_kwargs(rnn_module_kwargs, "module"),
                **hypothesis.prefix_kwargs(optimizer_kwargs, 'optimizer'),
                **hypothesis.prefix_kwargs(iter_train_kwargs, 'iterator_train'),
                **hypothesis.prefix_kwargs(iter_valid_kwargs, 'iterator_valid')
            },
            transformer=None,
        )

    def preprocess(self, x_train, x_test, y_train):
        x_train = x_train[['idxs', 'idx_lens']].values
        x_test = x_test[['idxs', 'idx_lens']].values
        y_train = y_train.cat.codes.values
        return x_train, x_test, y_train

    def generate_batch(self, batch):
        seq_tensors = [torch.tensor(o[0][0], dtype=torch.long, device=self.device) for o in batch]
        padded_idxs = nn.utils.rnn.pad_sequence(seq_tensors, batch_first=True, padding_value=0)
        seq_lens = torch.tensor([o[0][1] for o in batch], dtype=torch.long, device=self.device)
        x = {'idxs': padded_idxs, 'seq_lens': seq_lens}
        y = torch.tensor([o[1] for o in batch], dtype=torch.long, device=self.device)
        return x, y

    def _basic_hyper_dists(self):
        dist = {
            'optimizer__lr': loguniform(a=0.0001, b=0.01, base=10),
            'iterator_train__batch_size': [16, 32, 64, 128, 256, 512],
            'module__embeddings': randomnewinstance(
                partial(torch.clone, self.embeddings),
                *[partial(torch.randn, self.embeddings.shape[0], n) for n in [8, 16, 32, 64]]
            ),
            'freeze_embedding': [True, False],
            'bidirectional': [True, False],
            'recurrent_depth': randint(1, 4),
            'recurrent_features': [64, 128, 256, 512],
            'fc_hidden_features': [64, 128, 256, 512],
            'fc_hidden_depth': [0, 1, 2],
            'dropout': uniform(),
        }
        return dist


class MeanEmbeddingHypothesis(hypothesis.AbstractNNHypothesis):

    def __init__(self, hyper_search_strat, hyper_search_kwargs, embeddings, device):

        self.device = device
        self.embeddings = embeddings
        super().__init__(
            mode='classification',
            hyper_search_strat=hyper_search_strat,
            hyper_search_kwargs=hyper_search_kwargs,
            estimator_kwargs={
                'module': nets.FastText,
                'module__embeddings': embeddings,
                'device': self.device,
                'iterator_train__collate_fn': self.generate_batch,
                'iterator_valid__collate_fn': self.generate_batch,
                **estimator_kwargs,
                **hypothesis.prefix_kwargs(mean_embedding_module_kwargs, "module"),
                **hypothesis.prefix_kwargs(optimizer_kwargs, 'optimizer'),
                **hypothesis.prefix_kwargs(iter_train_kwargs, 'iterator_train'),
                **hypothesis.prefix_kwargs(iter_valid_kwargs, 'iterator_valid')
            },
            transformer=None,
        )

    def preprocess(self, x_train, x_test, y_train):
        x_train = x_train[['idxs', 'idx_lens']].values
        x_test = x_test[['idxs', 'idx_lens']].values
        y_train = y_train.cat.codes.values
        return x_train, x_test, y_train

    def generate_batch(self, batch):
        idxs = [torch.tensor(o[0][0], dtype=torch.long, device=self.device) for o in batch]
        idxs = torch.cat(tensors=idxs, dim=0)
        offsets = torch.tensor(([0] + [o[0][1] for o in batch])[:-1]).cumsum(dim=0)
        x = {'idxs': idxs, 'offsets': offsets}
        y = torch.tensor([o[1] for o in batch], dtype=torch.long, device=self.device)
        return x, y

    def _basic_hyper_dists(self):
        dist = {
            'optimizer__lr': loguniform(a=0.0001, b=0.01, base=10),
            'iterator_train__batch_size': [16, 32, 64, 128, 256, 512],
            'module__embeddings': randomnewinstance(
                partial(torch.clone, self.embeddings),
                *[partial(torch.randn, self.embeddings.shape[0], n) for n in [8, 16, 32, 64]]
            ),
            'module__freeze_embedding': [True, False],
            'module__fc_hidden_features': [32, 64, 128, 256, 512],
            'module__fc_hidden_depth': [0, 1, 2],
            'module__dropout': uniform(),
        }
        return hypothesis.prefix_kwargs(kwargs=dist, prefix='estimator')
