import gensim.utils
import numpy as np
import pandas as pd
import re
import skorch
import torch
import torch.nn as nn

from sklearn.model_selection import cross_val_score

# USER NEEDS
# Train an RNN machine using glove embeddings and estimate performance of model
# Make predictions on test set

# Read the data from disk
# Tokenize the text
# Create a mapping from word to a unique integer
# Convert the text into lists of integers
# Load the data in whatever format your deep learning framework requires
# Pad the text so that all the sequences are the same length, so you can process them in batch


# PARAMS
BATCH_SIZE = 64


# SET THE DEVICE
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# GET EMBEDDING AND INDEXES
#   Load embedding vectors
filename = '/Users/gregwalsh/Downloads/GoogleNews-vectors-negative300.bin'
keyed_vectors = gensim.models.KeyedVectors.load_word2vec_format(fname=filename, binary=True)
#   Load data
df_train = pd.read_csv(filepath_or_buffer='data/train.csv', delimiter=',')
df_test = pd.read_csv(filepath_or_buffer='data/test.csv', delimiter=',')
df_all = pd.concat(objs=[df_train, df_test], axis=0, keys=['train', 'test'], sort=True)
#   Tokenize text
df_all['tokens'] = df_all['text'].apply(func=lambda text: re.findall(r"[A-Za-z\-]+'[A-Za-z]+|[A-Za-z]+", text))
#   Get embedding indexes
get = keyed_vectors.vocab.get
df_all['idxs'] = df_all['tokens'].apply(func=lambda ts: [get(t).index if get(t) else None for t in ts])
#   Print matched tokenization proportion
unmatched_count = sum(df_all['idxs'].apply(func=lambda idxs: idxs.count(None)))
total_count = sum(df_all['idxs'].apply(func=lambda idxs: len(idxs)))
print('{f} of words matched'.format(f=(total_count - unmatched_count) / total_count))
#   Exclude unmatched tokens
df_all['idxs'] = df_all['idxs'].apply(
    func=lambda xs: torch.tensor(data=[x for x in xs if x], dtype=torch.long, device=device)
)
#   Create minimal set of embeddings and clean up
unique_indexes = list(set(idx for sample in df_all['idxs'] for idx in sample))
old_to_new_map = {old_idx: new_idx for new_idx, old_idx in enumerate(unique_indexes)}
sorted_embedding_indexes = [old_idx for old_idx, _ in sorted(old_to_new_map.items(), key=lambda x: x[1])]
embeddings = torch.tensor(data=np.take(keyed_vectors.vectors, [sorted_embedding_indexes], 0), dtype=torch.float16, device=device)
# del keyed_vectors
#   Reindex the tokens
df_all['idxs'] = df_all['idxs'].apply(func=lambda xs: [old_to_new_map.get(x) for x in xs])
#   Trim very long sequences
df_all['idx_lens'] = df_all['idxs'].apply(func=lambda idxs: len(idxs))
len_nth_percentile = int(df_all['idx_lens'].quantile(0.999))
df_all['idxs'] = df_all['idxs'].apply(func=lambda idxs: idxs[:len_nth_percentile])
df_all['idx_lens'] = df_all['idx_lens'].clip(upper=len_nth_percentile)


# CREATE LOADERS
#   Pad sequences to allow batch embedding (padding deleted in 'forward()' before recurrent layer)
x_train = nn.utils.rnn.pad_sequence(sequences=df_all['idxs']['train'], padding_value=0, batch_first=True)
x_test = nn.utils.rnn.pad_sequence(sequences=df_all['idxs']['test'], padding_value=0, batch_first=True)
#   Append len information to x so it we can create a packed seq in forward method of net
x_train = torch.cat(tensors=[x_train, torch.tensor(data=df_all['idx_lens']['train'], device=device).view(-1, 1)], dim=1)
x_test = torch.cat(tensors=[x_test, torch.tensor(data=df_all['idx_lens']['test'], device=device).view(-1, 1)], dim=1)
#   Create train target
y_train = torch.tensor(data=pd.Categorical(df_train['author']).codes, dtype=torch.long, device=device)


# DEFINE MODEL
class RNN(nn.Module):

    def __init__(self):
        super().__init__()
        self.num_layers = 1
        self.num_directions = 1
        self.embedding = nn.Embedding.from_pretrained(embeddings=embeddings)
        self.recurrent = nn.GRU(input_size=300, hidden_size=256, num_layers=self.num_layers,
                                bidirectional=self.num_directions == 2, batch_first=True)
        self.linear = nn.Linear(in_features=256 * self.num_directions, out_features=3)

    def forward(self, x):
        seq_lens = x[:, -1].detach().to(device='cpu')
        x = self.embedding(x[:, :-1])
        x = nn.utils.rnn.pack_padded_sequence(input=x, lengths=seq_lens, enforce_sorted=False, batch_first=True)
        _, h = self.recurrent(x)
        if self.num_directions == 1 and self.num_layers == 1:
            h = h.view(BATCH_SIZE, -1)
        else:
            h = torch.transpose(h[::self.num_layers], 0, 1).contiguous().view(BATCH_SIZE, -1)  # Get act's of last layer
        return self.linear(h)


# CREATE CLF
split = skorch.dataset.CVSplit(cv=5, stratified=True)
early_stopping = skorch.callbacks.EarlyStopping(
    monitor='valid_loss',
    patience=2,
    threshold=0.01,
    threshold_mode='rel',
    lower_is_better=True,
)
clf = skorch.NeuralNetClassifier(
    module=RNN,
    callbacks=[('early_stopping', early_stopping)],
    criterion=nn.CrossEntropyLoss,
    optimizer=torch.optim.Adam,
    train_split=split,
    iterator_train__shuffle=True,
    iterator_train__drop_last=True,
    iterator_valid__drop_last=True,
    batch_size=BATCH_SIZE,
    verbose=3
)

# clf.fit(X=x_train, y=y_train)
cross_val_score(estimator=clf, X=x_train.numpy(), y=y_train.numpy(), verbose=1, n_jobs=-1)


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
