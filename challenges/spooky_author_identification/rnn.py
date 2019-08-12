import gensim.utils
import operator
import pandas as pd
import re
import skorch
import torch
import torch.nn as nn

from sklearn.metrics import make_scorer, log_loss
from sklearn.model_selection import cross_validate, train_test_split
from skorch.helper import predefined_split
from skorch.dataset import Dataset
from scipy.special import softmax


# PARAMS
TARGET_COLUMNS = ["EAP", "HPL", "MWS"]


# SET THE DEVICE
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# LOAD EMBEDDINGS
#   Load embedding vectors
filename = '/Users/gregwalsh/Downloads/GoogleNews-vectors-negative300.bin'
keyed_vectors = gensim.models.KeyedVectors.load_word2vec_format(fname=filename, binary=True)


# LOAD DATA AND GET EMBEDDING INDEXES
#   Load data
df_train = pd.read_csv(filepath_or_buffer='data/train.csv', delimiter=',')
df_test = pd.read_csv(filepath_or_buffer='data/test.csv', delimiter=',')
df_train, df_val = train_test_split(df_train, stratify=df_train['author'], test_size=0.2)
df_train.reset_index(drop=True, inplace=True)
df_val.reset_index(drop=True, inplace=True)
df_all = pd.concat(objs=[df_train, df_val, df_test], axis=0, keys=['train', 'val', 'test'], sort=True)
df_all['author'] = pd.Categorical(df_all['author'], categories=TARGET_COLUMNS, ordered=True)

#   Tokenize text
df_all['tokens'] = df_all['text'].apply(func=lambda t: re.findall(r"[A-Za-z\-]+'[A-Za-z]+|[A-Za-z]+", t))

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
sorted_embedding_indexes = [old_idx for old_idx, _ in sorted(old_to_new_idxs.items(), key=operator.itemgetter(1))]
embeddings = torch.tensor(
    data=keyed_vectors.vectors[sorted_embedding_indexes, :], dtype=torch.float32, device=device
)
# del keyed_vectors

#   Update the indexes to the new embedding and pack in a tensor
df_all['idxs'] = df_all['idxs'].apply(
    func=lambda idxs: torch.tensor([old_to_new_idxs.get(idx) for idx in idxs], dtype=torch.long, device=device)
)


# CREATE LOADERS
#   Pad sequences to allow batch embedding (padding deleted in 'forward()' before recurrent layer)
xs = {}
ys = {}
for partition in ['train', 'val', 'test']:
    xs[partition] = nn.utils.rnn.pad_sequence(sequences=df_all['idxs'][partition], padding_value=0, batch_first=True)
    seq_lens = torch.tensor(data=df_all['idx_lens'][partition], device=device).view(-1, 1)
    xs[partition] = torch.cat(tensors=[xs[partition], seq_lens], dim=1)
for partition in ['train', 'val']:
    ys[partition] = torch.tensor(data=df_all['author'][partition].cat.codes, dtype=torch.long, device=device)


# DEFINE MODEL
class RNN(nn.Module):

    def __init__(self):
        super().__init__()
        self.num_layers = 1
        self.num_directions = 1
        self.hidden_size = 128
        self.embedding = nn.Embedding.from_pretrained(embeddings=embeddings)
        self.recurrent = nn.GRU(
            input_size=embeddings.shape[1], hidden_size=self.hidden_size, num_layers=self.num_layers,
            bidirectional=self.num_directions == 2, batch_first=True
        )
        self.linear = nn.Linear(in_features=self.hidden_size * self.num_directions, out_features=3)

    def forward(self, x):
        seq_lens = x[:, -1].detach().to(device='cpu')
        x = self.embedding(x[:, :-1])
        x = nn.utils.rnn.pack_padded_sequence(input=x, lengths=seq_lens, enforce_sorted=False, batch_first=True)
        _, h = self.recurrent(x)
        if self.num_directions != 1 or self.num_layers != 1:
            h = torch.transpose(h[::self.num_layers], 0, 1).contiguous()
        h = h.view(-1, self.hidden_size * self.num_directions)
        return self.linear(h)


# CREATE CLF
split = skorch.dataset.CVSplit(cv=5, stratified=True)
early_stopping = skorch.callbacks.EarlyStopping(
    monitor='valid_loss',
    patience=2,
    threshold=0,
    threshold_mode='rel',
    lower_is_better=True,
)
clf = skorch.NeuralNetClassifier(
    module=RNN,
    device=device,
    callbacks=[('early_stopping', early_stopping)],
    criterion=nn.CrossEntropyLoss,
    optimizer=torch.optim.Adam,
    train_split=predefined_split(Dataset(X=xs['val'], y=ys['val'])),
    iterator_train__shuffle=True,
    iterator_train__drop_last=True,
    iterator_valid__drop_last=False,
    iterator_train__batch_size=128,
    iterator_valid__batch_size=-1,  # use all examples
    verbose=1
)


# FIT MODEL
clf.fit(X=xs['train'], y=ys['train'])


# WRITE OUT PREDICTIONS TO FILE
df_pred = pd.DataFrame(data=clf.predict_proba(X=xs['test']), columns=TARGET_COLUMNS)
df_pred[TARGET_COLUMNS] = softmax(x=df_pred[TARGET_COLUMNS], axis=1)
df_pred['id'] = df_all['id']['test']
df_pred[['id'] + TARGET_COLUMNS].to_csv(path_or_buf='predictions/spooky_submission.csv', index=False)


# CROSS VALIDATE
def negative_log_loss(y_true, y_pred_raw):
    y_pred = softmax(x=y_pred_raw, axis=1)
    return log_loss(y_true=y_true, y_pred=y_pred)


nll_scorer = make_scorer(score_func=negative_log_loss, greater_is_better=False, needs_proba=True)

cv = cross_validate(
    estimator=clf,
    X=xs['train'].numpy(),
    y=ys['train'].numpy(),
    cv=3,
    scoring={'log loss': nll_scorer, 'accuracy': 'accuracy'},
    verbose=0,
    n_jobs=1,
)


########################################################################################################################


# pattern = re.compile('[\w\-]+')
# with open(file='data/test.csv', mode='r') as infile:
#     reader = csv.reader(infile, delimiter=',', quotechar='"')
#     with open(file='data/test_idxs.csv', mode='w') as outfile:
#         for line in reader:
#             words = re.findall(pattern=pattern, string=line[0])
#             outfile.write([keyed_vectors.x(word) for word in words])
#
# # Read examples
# train_examples = data.Example.fromCSV(data='data/train.csv', fields=[
#     ('text', data.Field(sequential=True, lower=False, include_lengths=True, batch_first=True)),
#     ('label', data.Field(sequential=False))
#     ])
#
# test_examples = data.Example.fromCSV(data='data/test.csv')
#
# # set up fields
# train = data.Dataset(examples='data/test.csv', fields=[
#     ('text', data.Field(sequential=True, lower=False, include_lengths=True, batch_first=True)),
#     ('label', data.Field(sequential=False))
#     ]
# )
#
# # make splits for data
# # train, test = datasets.IMDB.splits(TEXT, LABEL)
#
# # build the vocabulary
# keyed_vectors = gensim.models.KeyedVectors.load_word2vec_format('/Users/gregwalsh/Downloads')
# vectors = torch.FloatTensor(keyed_vectors.vectors)
# # TEXT.build_vocab(train, vectors=vectors)
# # LABEL.build_vocab(train)
#
# # make iterator for splits
# train_iter, test_iter = data.BucketIterator.splits(
#     (train, test), batch_size=3, device=0)
#
#
#
#
# # use torchtext to define the dataset field containing text
# text_field = dat.Field(sequential=True)
# target_field = dat.Field(sequential=True)
#
# # load your dataset using torchtext, e.g.
# train = data.Dataset(examples='data/test.csv', fields=[('text', text_field), ...])
#
# # build vocabulary
# text_field.build_vocab(dataset)
#
# # I use embeddings created with
# # model = gensim.models.Word2Vec(...)
# # model.wv.save_word2vec_format(path_to_embeddings_file)
#
# # load embeddings using torchtext
# vectors = dat.Vectors('/Users/gregwalsh/Downloads') # file created by gensim
# text_field.vocab.set_vectors(vectors.stoi, vectors.vectors, vectors.dim)
#
# # when defining your network you can then use the method mentioned by blue-phoenox
# embedding = nn.Embedding.from_pretrained(torch.FloatTensor(text_field.vocab.vectors))
#
#
# # GET EMBEDDING VECTORS
# keyed_vectors = gensim.models.KeyedVectors.load_word2vec_format('/Users/gregwalsh/Downloads')
# # weights = torch.FloatTensor(keyed_vectors.vectors)
#
# # PARSE THE DATA
# pattern = re.compile('[\w\-]+')
# with open(file='data/test.csv', mode='r') as infile:
#     reader = csv.reader(infile, delimiter=',', quotechar='"')
#     with open(file='data/test_idxs.csv', mode='w') as outfile:
#         for line in reader:
#             words = re.findall(pattern=pattern, string=line[0])
#             outfile.write([keyed_vectors.x(word) for word in words])
#
#
#
#
# CONTEXT_SIZE = 2  # num tokens to left and right of target
#
# # By deriving a set from `raw_text`, we deduplicate the array
# vocab = set(raw_text)
# vocab_size = len(vocab)
#
# word_to_ix = {word: i for i, word in enumerate(vocab)}
# data = []
# for i in range(2, len(raw_text) - 2):
#     context = [raw_text[i - 2], raw_text[i - 1],
#                raw_text[i + 1], raw_text[i + 2]]
#     target = raw_text[i]
#     data.append((context, target))
# print(data[:5])
#
# class CBOW(nn.Module):
#
#     def __init__(self, vocab_size, embedding_dim, context_size):
#         super(CBOW, self).__init__()
#         self.embeddings = nn.Embedding(vocab_size, embedding_dim)
#         self.linear1 = nn.Linear(embedding_dim, 128)
#         self.linear2 = nn.Linear(128, vocab_size)
#
#     def forward(self, inputs):
#         # print('input:', inputs)
#         embeds = self.embeddings(inputs).sum(dim=0)
#         # print('embeds:', embeds)
#         out_1 = F.relu(self.linear1(embeds))
#         out_2 = self.linear2(out_1)
#         # print('out_2:', out_2)
#         log_probs = F.log_softmax(out_2, dim=1)
#         # print('log_probs:', log_probs)
#         return log_probs
#
# # create your model and train.  here are some functions to help you make
# # the data ready for use by your module
#
# str.maketrans()
# def make_context_vector(context, word_to_ix):
#     idxs = [word_to_ix[w] for w in context]
#     return torch.tensor(idxs, dtype=torch.long)
#
# losses = []
# loss_function = nn.NLLLoss()
# model = CBOW(vocab_size, EMBEDDING_DIM, CONTEXT_SIZE)
# optimizer = optim.SGD(model.parameters(), lr=0.001)
#
#
