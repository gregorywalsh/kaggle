from skorch import NeuralNetClassifier


CONTEXT_SIZE = 2  # num tokens to left and right of target

# By deriving a set from `raw_text`, we deduplicate the array
vocab = set(raw_text)
vocab_size = len(vocab)

word_to_ix = {word: i for i, word in enumerate(vocab)}
data = []
for i in range(2, len(raw_text) - 2):
    context = [raw_text[i - 2], raw_text[i - 1],
               raw_text[i + 1], raw_text[i + 2]]
    target = raw_text[i]
    data.append((context, target))
print(data[:5])

class CBOW(nn.Module):

    def __init__(self, vocab_size, embedding_dim, context_size):
        super(CBOW, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.linear1 = nn.Linear(embedding_dim, 128)
        self.linear2 = nn.Linear(128, vocab_size)

    def forward(self, inputs):
        # print('input:', inputs)
        embeds = self.embeddings(inputs).sum(dim=0)
        # print('embeds:', embeds)
        out_1 = F.relu(self.linear1(embeds))
        out_2 = self.linear2(out_1)
        # print('out_2:', out_2)
        log_probs = F.log_softmax(out_2, dim=1)
        # print('log_probs:', log_probs)
        return log_probs

# create your model and train.  here are some functions to help you make
# the data ready for use by your module

str.maketrans()
def make_context_vector(context, word_to_ix):
    idxs = [word_to_ix[w] for w in context]
    return torch.tensor(idxs, dtype=torch.long)

losses = []
loss_function = nn.NLLLoss()
model = CBOW(vocab_size, EMBEDDING_DIM, CONTEXT_SIZE)
optimizer = optim.SGD(model.parameters(), lr=0.001)