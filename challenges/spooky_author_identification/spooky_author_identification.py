import challenges.spooky_author_identification.hypotheses as hyps
import gensim
import pandas as pd
import pickle
import re
import torch

from datareader import DataReader
from experiment import Experiment
from functools import partial
from operator import itemgetter
from sklearn.metrics import make_scorer
from sklearn.model_selection import RandomizedSearchCV, RepeatedStratifiedKFold
from sklearn.base import clone
from scipy.special import softmax
from temperature_scaling import ModelWithTemperature
from torch.utils.data import DataLoader

# CONSTANTS
DIR = 'challenges/spooky_author_identification'


# VARIABLES
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
cv_splits = 2
cv_repeats = 1


# # LOAD DATA
# #   Load data
# data_reader = DataReader(config_fp='{}/data/config.yml'.format(DIR))
# df_train = data_reader.load_from_csv(
#     fp='{}/data/train.csv'.format(DIR),
#     validate_col_names=True,
#     is_test=False,
#     append_vartype=True
# )
# df_test = data_reader.load_from_csv(
#     fp='{}/data/test.csv'.format(DIR),
#     validate_col_names=True,
#     is_test=True,
#     append_vartype=True
# )
#
# # EMBEDDED EXPERIMENTS
# #   Load embedding vectors
# filename = '/Users/gregwalsh/Downloads/GoogleNews-vectors-negative300.bin'
# keyed_vectors = gensim.models.KeyedVectors.load_word2vec_format(fname=filename, binary=True)
#
# #   Combine dataframes
# df_all = pd.concat(objs=[df_train, df_test], axis=0, keys=['train', 'test'], sort=True)
# df_all['author|categorical'] = pd.Categorical(df_all['author|categorical'], categories=["EAP", "HPL", "MWS"], ordered=True)
#
# #   Tokenize the text then get embedding indexes for tokens
# df_all['tokens'] = df_all['text|string'].apply(func=lambda t: re.findall(r"[A-Za-z\-]+'[A-Za-z]+|[A-Za-z]+", t))
# get = keyed_vectors.vocab.get
# df_all['idxs'] = df_all['tokens'].apply(func=lambda ts: [get(t).index if get(t) else None for t in ts])
#
# #   Report on matched tokenization proportion
# unmatched_count = sum(df_all['idxs'].apply(func=lambda idxs: idxs.count(None)))
# total_count = sum(df_all['idxs'].apply(func=lambda idxs: len(idxs)))
# print('{f} of words matched'.format(f=(total_count - unmatched_count) / total_count))
#
#   Exclude unmatched tokens, trim very long seqs, and store lengths for seq packing later
# df_all['idxs'] = df_all['idxs'].apply(func=lambda idxs: [idx for idx in idxs if idx])
# df_all['idx_lens'] = df_all['idxs'].apply(func=lambda idxs: len(idxs))
# len_nth_percentile = int(df_all['idx_lens'].quantile(0.999))
# df_all['idxs'] = df_all['idxs'].apply(func=lambda idxs: idxs[:len_nth_percentile])
# df_all['idx_lens'] = df_all['idx_lens'].clip(upper=len_nth_percentile)
#
# #   Create minimal set of new embeddings and clean up memory
# unique_old_indexes = list(set(idx for sample in df_all['idxs'] for idx in sample))
# idxs_map = {old_idx: new_idx for new_idx, old_idx in enumerate(unique_old_indexes)}
# sorted_embedding_indexes = [old_idx for old_idx, _ in sorted(idxs_map.items(), key=itemgetter(1))]
# minimal_keyed_vectors = torch.tensor(data=keyed_vectors.vectors[sorted_embedding_indexes, :], dtype=torch.float32)
# del keyed_vectors
#
# #   Update the indexes to the new embedding and pack in a tensor
# df_all['idxs'] = df_all['idxs'].apply(func=lambda idxs: [idxs_map.get(idx) for idx in idxs])
#
# #   Save the processed data end minimal embedding
# pickle.dump(df_all, open('df_all.pkl', 'wb'))
# pickle.dump(minimal_keyed_vectors, open('minimal_keyed_vectors.pkl', 'wb'))

#   Load the processed data
df_all = pickle.load(open('df_all.pkl', 'rb'))
minimal_keyed_vectors = pickle.load(open('minimal_keyed_vectors.pkl', 'rb'))

#   Define hypotheses
hyper_search_strat = RandomizedSearchCV
hyper_search_kwargs = {
    'cv': RepeatedStratifiedKFold(n_splits=cv_splits, n_repeats=cv_repeats),
    'scoring': 'neg_log_loss',
    'error_score': 'raise',
    'refit': True,
    'n_jobs': 1
}

hypotheses = [
    ('rnn', hyps.RNNHypothesis(
        hyper_search_strat=hyper_search_strat,
        hyper_search_kwargs=hyper_search_kwargs,
        embeddings=minimal_keyed_vectors,
        device=device
    ))
]

# BUILD AND RUN EXPERIMENT
experiment = Experiment(
    x_train=df_all[['id|unique', 'idxs', 'idx_lens']].loc['train'],
    x_test=df_all[['id|unique', 'idxs', 'idx_lens']].loc['test'],
    y_train=df_all['author|categorical'].loc['train'],
    hypotheses=hypotheses
)
experiment.run(
    num_hyper_samples=0,
    directory=DIR
)

# FIT MODEL
clf.fit(X=xs['train'], y=ys['train'])
clf.initialize()
clf.load_params('{c}/models/rnn/params.pt'.format(c=DIR))


# CALIBRATE PROBABILITIES
clf_clone = clone(clf)
cal_model = ModelWithTemperature(model=clf_clone.module_, device=device)
val_loader = DataLoader(val_dataset, batch_size=128, shuffle=True, drop_last=True)
clf_clone.module_ = cal_model.set_temperature(valid_loader=val_loader)


# WRITE OUT PREDICTIONS TO FILE
df_pred = pd.DataFrame(data=clf_clone.predict_proba(X=xs['test']), columns=TARGET_COLUMNS)
df_pred[TARGET_COLUMNS] = softmax(x=df_pred[TARGET_COLUMNS], axis=1)
df_pred['id'] = df_all['id']['test']
df_pred[['id'] + TARGET_COLUMNS].to_csv(path_or_buf='{c}/predictions/submission_temp.csv'.format(c=DIR), index=False)


# CROSS VALIDATE
def negative_log_loss(y_true, y_pred_logits):
    log_loss = torch.CrossEntroyLoss()
    return log_loss(torch.tensor(y_true), torch.long(y_pred_logits))


nll_scorer = make_scorer(score_func=negative_log_loss, greater_is_better=False, needs_proba=True)
