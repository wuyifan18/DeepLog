# berts.py

import torch
from torch.nn import functional as F

class PositionEncoding(torch.nn.Module):
    def __init__(self, hidden_size, max_len=5000, name='PositionEncoding'):
        # PositionEncoding from https://github.com/oliverguhr/transformer-time-series-prediction/blob/570d39bc0bbd61c823626ec9ca8bb5696c068187/transformer-singlestep.py#L25
        super(PositionEncoding, self).__init__()
        pe = torch.zeros(max_len, hidden_size)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_team = torch.exp(torch.arange(0, hidden_size, 2).float() * (-math.log(10000.0) / hidden_size))
        pe[:, 0::2] = torch.sin(position * div_team)
        pe[:, 1::2] = torch.cos(position * div_team)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, inputs):
        return inputs + self.pe[:inptus.size(0), :]


class LogClassifier(torch.nn.Module):
    def __init__(self, input_hidden_size, hidden_size, num_class, num_layer=2, dropout=0.0, name='LogClassifier'):
        super(LogClassifier, self).__init__()

        # initialize hidden layers
        self.classifier = []
        for _ in range(num_layer):
            self.classifier.extend([
                torch.nn.Linear(input_hidden_size, hidden_size),
                torch.nn.ReLU(),
                torch.nn.Dropout(dropout)
            ])
            input_hidden_size = hidden_size

        # final layer
        self.classifier.append(torch.nn.Linear(hidden_size, num_class))

        # convert list ot ModuleList
        self.classifier = torch.nn.ModuleList(self.classifier)

    def forward(self, inputs):
        return self.classifier(inputs)


class LogTransformer(torch.nn.Module):
    # A Transformer-based encoder for abnormally detection on logs
    def __init__(self, num_class, vocab_size, embed_size, hidden_size, num_layer, num_head,
                 dropout, decoder_hidden_size, name='LogTransformer'):
        super(LogTransformer, self).__init__()

        # initialize embedding
        self.log_embed = torch.nn.Embedding(num_embeddings=vocab_size, embedding_dim=embed_size)

        # initialize encoder
        self.pos_encoder = PositionEncoding(hidden_size)
        self.transformer_layer = torch.nn.TransformerEncoderLayer(d_model=hidden_size,
                                                                  nhead=num_head,
                                                                  dropout=dropout)
        self.encoder = torch.nn.TransformerEncoder(self.transformer_layer,
                                                   num_layers=num_layer)

        # initialize decoder
        self.decoder = LogClassifier(input_hidden_size=hidden_size,
                                    hidden_size=decoder_hidden_size,
                                    num_class=num_class,
                                    dropout=dropout)

    def predict(self, inputs):
        # forward
        outputs = self.forward()

        # softmax
        outputs = F.softmax(outputs, dim=-1)
        return outputs

    def compute_loss(self, outputs, labels):
        # Function to compute loss
        return F.cross_entropy(outputs, labels)

    def _forward(self, inputs):
        # embed logs-positions and logs
        pos_features = self.pos_encoder(inputs['pos_inputs'])
        log_features = self.log_embed(inputs['log_inputs'])
        features = torch.cat([pos_features, log_features], dim=-1)

        # encode
        features = self.encoder(features, inputs)
        # predict next log entries
        outputs = self.decoder(features)

        return outputs

    def forward(self, inputs, labels):
        # forward step
        outputs = self._forward(inputs)

        # compute loss
        loss = self.compute_loss(outputs, labels)
        
        return loss
