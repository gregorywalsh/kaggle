import pandas as pd
import re
import torch

from hypothesis import NNHypothesis
from operator import itemgetter
from nets import GRUVarLenSeq
from sklearn.model_selection import train_test_split
from torch import nn


class RNN(NNHypothesis):

    def __init__(self, hyper_search_strat, hyper_search_kwargs, net_kwargs, directory):
        super().__init__(
            mode='classification',
            module=GRUVarLenSeq,
            use_gpu=True,
            checkpoint_dir='{}/checkpoints'.format(directory),
            hyper_search_strat=hyper_search_strat,
            hyper_search_kwargs=hyper_search_kwargs,
            transformer=None,
            additional_hyper_dists=None,
            net_kwargs=net_kwargs
        )

    def preprocess(self, x_train, x_test, y_train):
        #   Combine dataframes
        df_train = pd.concat(objs=[x_train, y_train], axis=1)
        df_all = pd.concat(objs=[df_train, x_test], axis=0, keys=['train', 'test'], sort=True)
        df_all['author|categorical'] = pd.Categorical(df_all['author|categorical'], categories=["EAP", "HPL", "MWS"], ordered=True)
        #   Tokenize text
        df_all['tokens'] = df_all['text|string'].apply(func=lambda t: re.findall(r"[A-Za-z\-]+'[A-Za-z]+|[A-Za-z]+", t))
        #   Get embedding indexes
        get = keyed_vectors.vocab.get
        df_all['idxs'] = df_all['tokens'].apply(func=lambda ts: [get(t).index if get(t) else None for t in ts])
        #   Print matched tokenization proportion
        unmatched_count = sum(df_all['idxs'].apply(func=lambda idxs: idxs.count(None)))
        total_count = sum(df_all['idxs'].apply(func=lambda idxs: len(idxs)))
        print('{f} of words matched'.format(f=(total_count - unmatched_count) / total_count))
        #   Exclude unmatched tokens
        df_all['idxs'] = df_all['idxs'].apply(func=lambda idxs: [idx for idx in idxs if idx])
        #   Trim very long seqs and store lengths for seq packing later
        df_all['idx_lens'] = df_all['idxs'].apply(func=lambda idxs: len(idxs))
        len_nth_percentile = int(df_all['idx_lens'].quantile(0.999))
        df_all['idxs'] = df_all['idxs'].apply(func=lambda idxs: idxs[:len_nth_percentile])
        df_all['idx_lens'] = df_all['idx_lens'].clip(upper=len_nth_percentile)
        df_all['idx_lens'] = df_all['idx_lens'].apply(lambda l: torch.tensor(data=l, device=device).view(-1, 1))

        #   Create minimal set of new embeddings and clean up
        unique_old_indexes = list(set(idx for sample in df_all['idxs'] for idx in sample))
        old_to_new_idxs = {old_idx: new_idx for new_idx, old_idx in enumerate(unique_old_indexes)}
        sorted_embedding_indexes = [old_idx for old_idx, _ in sorted(old_to_new_idxs.items(), key=itemgetter(1))]
        embeddings = torch.tensor(
            data=keyed_vectors.vectors[sorted_embedding_indexes, :], dtype=torch.float32, device=device
        )
        # del keyed_vectors

        #   Update the indexes to the new embedding and pack in a tensor
        df_all['idxs'] = df_all['idxs'].apply(
            func=lambda idxs: torch.tensor([old_to_new_idxs.get(idx) for idx in idxs], dtype=torch.long, device=device)
        )
        # CREATE LOADERS
        df_train, df_val = train_test_split(df_train, stratify=df_train['author|categorical'], test_size=0.2)
        for df in [df_train, df_val]:
            df.reset_index(drop=True, inplace=True)
        #   Pad sequences to allow batch embedding (padding deleted in 'forward()' before recurrent layer)
        xs = {}
        ys = {}
        for partition in ['train', 'val', 'test']:
            xs[partition] = nn.utils.rnn.pad_sequence(sequences=df_all['idxs'][partition], padding_value=0, batch_first=True)
            seq_lens = torch.tensor(data=df_all['idx_lens'][partition], device=device).view(-1, 1)
            xs[partition] = torch.cat(tensors=[xs[partition], seq_lens], dim=1)
        for partition in ['train', 'val']:
            ys[partition] = torch.tensor(data=df_all['author|categorical'][partition].cat.codes, dtype=torch.long, device=device)