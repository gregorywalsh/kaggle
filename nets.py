import torch
import torch.nn as nn

class MLP(nn.Module):

    def __init__(self, in_features, hidden_features, hidden_depth, out_features, dropout, batch_norm):
        super().__init__()
        hidden_layers = []
        for i in range(hidden_depth):
            layer = []
            if i == 0:
                layer.append(nn.Linear(in_features=in_features, out_features=hidden_features))
            else:
                layer.append(nn.Linear(in_features=hidden_features, out_features=hidden_features))
            layer.append(nn.ReLU())
            if batch_norm:
                layer.append(nn.BatchNorm1d(hidden_features))
            if dropout:
                layer.append(nn.Dropout(p=dropout))
            hidden_layers.append(nn.Sequential(*layer))
        self.hidden_layers = nn.Sequential(*hidden_layers) if hidden_layers else None
        self.out = nn.Linear(in_features=hidden_features if hidden_layers else in_features, out_features=out_features)

    def forward(self, x):
        if self.hidden_layers:
            x = self.hidden_layers(x)
        x = self.out(x)
        return x


class RNN(nn.Module):

    def __init__(self, embeddings, freeze_embedding, recurrent_depth, bidirectional, recurrent_features,
                 fc_hidden_features, fc_hidden_depth, out_features, dropout=0.5, batch_norm=True):
        super().__init__()
        self.recurrent_depth = recurrent_depth
        self.bidirectional = bidirectional
        self.recurrent_features = recurrent_features
        self.embedding = nn.Embedding.from_pretrained(embeddings=embeddings, freeze=freeze_embedding)
        self.recurrent = nn.GRU(
            input_size=embeddings.shape[1],
            hidden_size=recurrent_features,
            num_layers=recurrent_depth,
            bidirectional=bidirectional,
            batch_first=True,
            dropout=dropout if recurrent_depth > 1 else 0
        )
        self.out = MLP(
            in_features=recurrent_features * self.bidirectional,
            hidden_features=fc_hidden_features,
            hidden_depth=fc_hidden_depth,
            out_features=out_features,
            dropout=dropout,
            batch_norm=batch_norm
        )

    def forward(self, idxs, seq_lens):
        x = self.embedding(idxs)
        x = nn.utils.rnn.pack_padded_sequence(input=x, lengths=seq_lens, batch_first=True, enforce_sorted=False)
        _, h = self.recurrent(x)
        if self.bidirectional or self.recurrent_depth != 1:
            h = torch.transpose(h[::self.recurrent_depth], 0, 1).contiguous()  # Flatten and transpose to batch first
        h = h.view(-1, self.recurrent_features * (2 if self.bidirectional else 1))
        x = self.out(h)
        return x


class FastText(nn.Module):
    """
    Model described here
    """

    def __init__(self, embeddings, freeze_embedding, fc_hidden_features, fc_hidden_depth, out_features, dropout=0.5,
                 batch_norm=True):
        super().__init__()
        self.embedding = nn.EmbeddingBag.from_pretrained(embeddings=embeddings, freeze=freeze_embedding)
        self.batch_norm = nn.BatchNorm1d(num_features=embeddings.shape[1])
        self.out = MLP(
            in_features=embeddings.shape[1],
            hidden_features=fc_hidden_features,
            hidden_depth=fc_hidden_depth,
            out_features=out_features,
            dropout=dropout,
            batch_norm=batch_norm
        )

    def forward(self, idxs, offsets):
        x = self.embedding(*[idxs, offsets])
        x = self.batch_norm(x)
        x = self.out(x)
        return x
