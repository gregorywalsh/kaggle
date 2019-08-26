import torch
import torch.nn as nn


class GRUVarLenSeq(nn.Module):

    def __init__(self, mode, embedding, num_layers, num_directions, hidden_size, dropout):
        super().__init__()
        self.embedding = embedding
        self.num_layers = num_layers
        self.num_directions = num_directions
        self.hidden_size = hidden_size
        self.dropout = dropout
        if isinstance(self.embedding, torch.Tensor):
            self.embedding = nn.Embedding.from_pretrained(embeddings=self.embedding)
        elif isinstance(self.embedding, int):
            self.embedding = nn.Embedding(num_embeddings=self.embedding)
        else:
            raise ValueError("'embedding' must be 2D tensor or int")
        self.recurrent = nn.GRU(
            input_size=self.embeddings.shape[1], hidden_size=self.hidden_size, num_layers=self.num_layers,
            bidirectional=self.num_directions == 2, batch_first=True, dropout=self.dropout
        )
        if mode == 'classification':
            self.linear = nn.Linear(in_features=self.hidden_size * self.num_directions, out_features=3)
        elif mode == 'regression':
            self.linear = nn.Linear(in_features=self.hidden_size * self.num_directions, out_features=3)
        else:
            raise ValueError('"mode" must be one of {"classification", "regression"}')

    def forward(self, x):
        seq_lens = x[:, -1].detach().to(device='cpu')
        x = self.embedding(x[:, :-1])
        x = nn.utils.rnn.pack_padded_sequence(input=x, lengths=seq_lens, enforce_sorted=False, batch_first=True)
        _, h = self.recurrent(x)
        if self.num_directions != 1 or self.num_layers != 1:
            h = torch.transpose(h[::self.num_layers], 0, 1).contiguous()
        h = h.view(-1, self.hidden_size * self.num_directions)
        return self.linear(h)
