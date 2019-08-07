# USER NEEDS
# Train an RNN machine using glove embeddings and estimate performance of model
# Make predictions on test set



# Read the data from disk
# Tokenize the text
# Create a mapping from word to a unique integer
# Convert the text into lists of integers
# Load the data in whatever format your deep learning framework requires
# Pad the text so that all the sequences are the same length, so you can process them in batch

import pandas as pd
import re
import gensim.utils
import torch
import torch.nn as nn
import torchtext.data as data
import torchtext.vocab as vocab
from skorch import NeuralNetClassifier
from skorch.dataset import CVSplit

# LOAD DATASETS
train_df = pd.read_csv(filepath_or_buffer='data/train.csv', delimiter=',')
test_df = pd.read_csv(filepath_or_buffer='data/test.csv', delimiter=',')


# MUNGE
author_dummies_df = pd.get_dummies(data=train_df['author'])
train_df = pd.concat(objs=[train_df[['id', 'text']], author_dummies_df], axis=1)
text_df = pd.concat(objs=[train_df['text'], test_df['text']], axis=0, keys=['train', 'test'])
tokens_df = text_df.apply(func=lambda text: re.findall(r"[A-Za-z\-]+'[A-Za-z]+|[A-Za-z]+", text))


# GET INDEXES
#   Load model
filename = '/Users/gregwalsh/Downloads/GoogleNews-vectors-negative300.bin'
keyed_vectors = gensim.models.KeyedVectors.load_word2vec_format(fname=filename, binary=True)
#   Tokenize text
get = keyed_vectors.vocab.get
idxs_df = tokens_df.apply(func=lambda tokens: [get(token).index if get(token) else None for token in tokens])
#   Print tokenization proportion
unmatched_count = sum(idxs_df.apply(func=lambda idxs: idxs.count(None)))
total_count = sum(idxs_df.apply(func=lambda idxs: len(idxs)))
print('{f} of words matched'.format(f=(total_count - unmatched_count) / total_count))
#   Exclude unmatched tokens
idxs_df = idxs_df.apply(func=lambda idxs: torch.LongTensor([idx for idx in idxs if idx]))


# GET EMBEDDING
embeddings = torch.FloatTensor(keyed_vectors.vectors)


# DEFINE MODEL
class RNN(nn.Module):

    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding.from_pretrained(embeddings=embeddings)
        self.recurrent = nn.GRU(input_size=300, hidden_size=256, num_layers=1, bidirectional=False)
        self.linear = nn.Linear(in_features=256, out_features=3)

    def forward(self, x):
        x = self.embedding(x)
        _, x = self.recurrent(x)
        return self.linear(x)


# CREATE CLF
split = CVSplit(cv=5, stratified=False)
clf = NeuralNetClassifier(
    module=RNN,
    criterion=nn.CrossEntropyLoss,
    optimizer=torch.optim.Adam,
    train_split=split,
    iterator_train__shuffle=True,
)

clf.fit(X=idxs_df['train'], y=author_dummies_df)


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
