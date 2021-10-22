# berts.py

import torch
from torch.nn import functional as F

class PositionEncoding(torch.nn.Module):
    def __init__(self, embed_size, name='PositionEncoding'):
        super(PositionEncoding, self).__init__()
        self.embed_size = embed_size

    def forward(self, inputs):
        return inputs


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
        return None

    def copmpute_loss(self, outputs, labels):
        # Function to compute loss
        return F.cross_entropy(outputs, labels)

    def forward(self, inputs, labels):
        # embed logs-positions and logs
        pos_features = self.pos_encoder(inputs['pos_inputs'])
        log_features = self.log_embed(inputs['log_inputs'])
        features = torch.cat([pos_features, log_features], dim=-1)

        # encode
        features = self.encoder(features, inputs)
        # predict next log entries
        outputs = self.decoder(features)

        # compute loss
        loss = self.compute_loss(outputs, labels)
        
        return loss
