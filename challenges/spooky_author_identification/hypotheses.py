import torch
import torch.nn as nn

from hypothesis import AbstractNNHypothesis
from nets import RNN, MeanEmbedding
from torch.utils.data import TensorDataset


def generate_batch(batch):
    seq_tensors = [torch.tensor(o[0][0], dtype=torch.long) for o in batch]
    padded_idxs = nn.utils.rnn.pad_sequence(seq_tensors, batch_first=True, padding_value=0)
    seq_lens = torch.tensor([o[0][1] for o in batch], dtype=torch.long)
    y = torch.tensor([o[1] for o in batch], dtype=torch.long)
    x = {'idxs': padded_idxs, 'seq_lens': seq_lens}
    return x, y


class RNNHypothesis(AbstractNNHypothesis):

    def __init__(self, hyper_search_strat, hyper_search_kwargs, embeddings, device):

        super().__init__(
            mode='classification',
            module=RNN,
            device=device,
            hyper_search_strat=hyper_search_strat,
            hyper_search_kwargs=hyper_search_kwargs,
            val_fraction=0.2,
            module_kwargs={
                'embeddings': embeddings,
                'freeze_embedding': False,
                'recurrent_depth': 1,
                'recurrent_directions': 1,
                'recurrent_features': 256,
                'fc_hidden_features': 256,
                'fc_hidden_depth': 1,
                'out_features': 3,
                'dropout': 0.5
            },
            iter_train_kwargs={
                'collate_fn': generate_batch,
                'batch_size': 128,
                'shuffle': True,
                'drop_last': True,
            },
            iter_valid_kwargs={
                'collate_fn': generate_batch,
                'batch_size': -1,
                'shuffle': False,
                'drop_last': False,
            },
            transformer=None,
            additional_hyper_dists=None,
        )

    def preprocess(self, x_train, x_test, y_train):
        x_train = x_train[['idxs', 'idx_lens']].values
        x_test = x_test[['idxs', 'idx_lens']].values
        y_train = y_train.cat.codes.values
        return x_train, x_test, y_train

    def generate_batch(self, batch):
        seq_tensors = [torch.tensor(o[0]['idxs'][0], dtype=torch.long, device=self._estimator.device) for o in batch]
        idxs = nn.utils.rnn.pad_sequence(seq_tensors, batch_first=True, padding_value=0).to(self._estimator.device)
        lens = torch.tensor([o[0]['idx_lens'] for o in batch], dtype=torch.long).view(-1, 1)
        labels = torch.tensor([o[1] for o in batch], dtype=torch.long)
        return idxs, lens, labels


class MeanEmbeddingHypothesis(AbstractNNHypothesis):

    @staticmethod
    def generate_batch(batch):
        label = torch.tensor([entry[0] for entry in batch])
        text = [entry[1] for entry in batch]
        offsets = [0] + [len(entry) for entry in text]
        offsets = torch.tensor(offsets[:-1]).cumsum(dim=0)
        text = torch.cat(text)
        return text, offsets, label

    def __init__(self, hyper_search_strat, hyper_search_kwargs, module_kwargs, device):
        super().__init__(
            mode='classification',
            module=MeanEmbedding,
            device=device,
            hyper_search_strat=hyper_search_strat,
            hyper_search_kwargs=hyper_search_kwargs,
            module_kwargs=module_kwargs,
            transformer=None,
            additional_hyper_dists=None,
        )

    def preprocess(self, x_train, x_test, y_train):
        x_train = x_train[['idxs', 'idx_lens']]
        x_test = x_test[['idxs', 'idx_lens']]
        y_train = torch.tensor(
            data=y_train['author|categorical'].cat.codes,
            dtype=torch.long,
            device=self._estimator.device.device
        )
        return x_train, x_test, y_train
