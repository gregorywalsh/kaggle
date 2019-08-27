import torch
import torch.nn as nn


class MLP(nn.Module):

    def __init__(self, in_features, hidden_features, hidden_depth, out_features, dropout):
        super().__init__()
        hidden_layers = []
        for i in range(hidden_depth):
            linear = nn.Linear(in_features=in_features if i == 0 else hidden_features, out_features=hidden_features)
            if dropout:
                hidden_layers.append(nn.Sequential(linear, nn.Dropout(p=dropout)))
            else:
                hidden_layers.append(linear)
        self.hidden_layers = nn.Sequential(*hidden_layers) if hidden_layers else None
        self.out = nn.Linear(in_features=hidden_features if hidden_layers else in_features, out_features=out_features)

    def forward(self, x):
        if self.hidden_layers:
            x = self.hidden_layers(x)
        x = self.out(x)
        return x


class RNN(nn.Module):

    def __init__(self, embeddings, freeze_embedding, recurrent_depth, recurrent_directions, recurrent_features,
                 fc_hidden_features, fc_hidden_depth, out_features, dropout):
        super().__init__()
        if recurrent_directions not in (1, 2):
            raise ValueError('Valid values for arg "directions" are {1, 2}')
        self.embedding = nn.Embedding.from_pretrained(embeddings=embeddings, freeze=freeze_embedding)
        self.recurrent = nn.GRU(
            input_size=self.embedding.embedding_dim,
            hidden_size=recurrent_features,
            num_layers=recurrent_depth,
            bidirectional=recurrent_directions,
            batch_first=True,
            dropout=dropout if recurrent_depth > 1 else 0
        )
        self.out = MLP(
            in_features=recurrent_features * recurrent_directions,
            hidden_features=fc_hidden_features,
            hidden_depth=fc_hidden_depth,
            out_features=out_features,
            dropout=dropout
        )

    def forward(self, idxs, seq_lens):
        x = self.embedding(idxs)
        x = nn.utils.rnn.pack_padded_sequence(input=x, lengths=seq_lens, batch_first=True, enforce_sorted=False)
        _, h = self.recurrent(x)
        if self.recurrent_directions != 1 or self.num_rec_layers != 1:
            h = torch.transpose(h[::self.num_rec_layers], 0, 1).contiguous()  # Flatten and transpose to batch first
        h = h.view(-1, self.recurrent_features * self.recurrent_directions)
        x = self.out(h)
        return x


class MeanEmbedding(nn.Module):

    def __init__(self, embeddings, freeze_embedding, fc_hidden_features, fc_hidden_depth, out_features, dropout):
        super().__init__()
        self.embedding = nn.EmbeddingBag.from_pretrained(embeddings=embeddings, freeze=freeze_embedding)
        self.out = MLP(
            in_features=self.embedding.embedding_dim,
            hidden_features=fc_hidden_features,
            hidden_depth=fc_hidden_depth,
            out_features=out_features,
            dropout=dropout
        )

    def forward(self, idxs, offsets):
        x = self.embedding(idxs, offsets)
        x = self.out(x)
        return x
