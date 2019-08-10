import pandas as pd
import re
import gensim.utils
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
from skorch import NeuralNetClassifier
from skorch.dataset import CVSplit
from sklearn.model_selection import train_test_split

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
#   Load model
filename = '/Users/gregwalsh/Downloads/GoogleNews-vectors-negative300.bin'
keyed_vectors = gensim.models.KeyedVectors.load_word2vec_format(fname=filename, binary=True)
embeddings = torch.FloatTensor(keyed_vectors.vectors)
#   Load data
train_df = pd.read_csv(filepath_or_buffer='data/train.csv', delimiter=',')
test_df = pd.read_csv(filepath_or_buffer='data/test.csv', delimiter=',')
#   Tokenize text
text_df = pd.concat(objs=[train_df['text'], test_df['text']], axis=0, keys=['train', 'test'])
tokens_df = text_df.apply(func=lambda text: re.findall(r"[A-Za-z\-]+'[A-Za-z]+|[A-Za-z]+", text))
get = keyed_vectors.vocab.get
idxs_df = tokens_df.apply(func=lambda tokens: [get(token).index if get(token) else None for token in tokens])
#   Print tokenization proportion
unmatched_count = sum(idxs_df.apply(func=lambda idxs: idxs.count(None)))
total_count = sum(idxs_df.apply(func=lambda idxs: len(idxs)))
print('{f} of words matched'.format(f=(total_count - unmatched_count) / total_count))
#   Exclude unmatched tokens
idxs_df = idxs_df.apply(func=lambda idxs: torch.LongTensor([idx for idx in idxs if idx]))
#   Trim very long sequences
idx_lens = idxs_df.apply(func=lambda idxs: len(idxs))
len_nth_percentile = int(idx_lens.quantile(0.999))
idxs_df = idxs_df.apply(func=lambda idxs: idxs[:len_nth_percentile])
idx_lens = idx_lens.clip(upper=len_nth_percentile)


# CREATE LOADERS
#   Pad sequences and pack into batch first tensor
x_train = pad_sequence(sequences=idxs_df['train'], padding_value=0, batch_first=True)  # Don't worry...
x_test = pad_sequence(sequences=idxs_df['test'], padding_value=0, batch_first=True)  # ...padding will be deleted later
#   Append len information to x so it we can create a packed seq in forward method of net
x_train = torch.cat(tensors=[x_train, torch.tensor(data=idx_lens['train']).view(-1, 1)], dim=1).to(device)
x_test = torch.cat(tensors=[x_test, torch.LongTensor(idx_lens['test']).view(-1, 1)], dim=1).to(device)
#   Create train target
y_train = torch.tensor(data=pd.Categorical(train_df['author']).codes, dtype=torch.long, device=device)


# # Pytorch train and test sets
# train = torch.utils.data.TensorDataset(x_train, y_train)
# test = torch.utils.data.TensorDataset(x_test)
#
# # data loader
# train_loader = torch.utils.data.DataLoader(train, batch_size=BATCH_SIZE, shuffle=True)
# test_loader = torch.utils.data.DataLoader(test, batch_size=BATCH_SIZE, shuffle=True)


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
        # print(x.shape)
        unpadded_lens = x[:, -1].clone().detach()
        # print(unpadded_lens.shape)
        x = self.embedding(x[:, :-1])
        x = pack_padded_sequence(input=x, lengths=unpadded_lens, enforce_sorted=False, batch_first=True)
        _, h = self.recurrent(x)
        # print(h.shape)
        if self.num_directions == 1 and self.num_layers == 1:
            h = h.view(BATCH_SIZE, -1)
        else:
            h = torch.transpose(h[::self.num_layers], 0, 1).contiguous().view(BATCH_SIZE, -1)  # Get act's of last layer
        # print(h.shape)
        return self.linear(h)


# CREATE CLF
split = CVSplit(cv=5, stratified=False)
clf = NeuralNetClassifier(
    module=RNN,
    criterion=nn.CrossEntropyLoss,
    optimizer=torch.optim.Adam,
    train_split=split,
    iterator_train__shuffle=True,
    iterator_train__batch_size=BATCH_SIZE,
    verbose=2
)

clf.fit(X=x_train, y=y_train)


for epoch in range(2):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 2000 == 1999:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0

print('Finished Training')






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
