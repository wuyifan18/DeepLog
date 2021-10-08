# berts.py

import torch


class PositionEncoding(torch.nn.Module):
    def __init__(self, embed_size, name='PositionEncoding'):
        super(PositionEncoding, self).__init__()
        self.embed_size = embed_size

    def forward(self, inputs):
        return inputs


class LogGenerator(torch.nn.Module):
    def __init__(self, input_hidden_size, hidden_size, num_class, num_layer=2, dropout=0.0, name='LogGenerator'):
        super(LogGenerator, self).__init__()

        # initialize hidden layers
        self.generator = [
            torch.nn.Linear(input_hidden_size, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout)
        ]
        self.generator.extend([
                                  torch.nn.Linear(hidden_size, hidden_size),
                                  torch.nn.ReLU(),
                                  torch.nn.Dropout(dropout)
                              ] * (num_layer - 1))

        # final layer
        self.generator.extend([
            torch.nn.Linear(hidden_size, num_class),
            torch.nn.Softmax()
        ])

        # convert list ot ModuleList
        self.generator = torch.nn.ModuleList(self.generator)

    def forward(self, inputs):
        return self.generator(inputs)


class LogTransformer(torch.nn.Module):
    # A Transformer-based encoder for abnormally detection on logs
    def __init__(self, num_class, encoder_hidden_size, decoder_hidden_size, num_layer, num_head, dropout,
                 name='LogTransformer'):
        super(LogTransformer, self).__init__()

        # initialize encoder
        self.pos_encoder = PositionEncoding(encoder_hidden_size)
        self.transformer_layer = torch.nn.TransformerEncoderLayer(d_model=encoder_hidden_size,
                                                                  nhead=num_head,
                                                                  dropout=dropout)
        self.encoder = torch.nn.TransformerEncoder(self.transformer_layer,
                                                   num_layers=num_layer)

        # initialize decoder
        self.decoder = LogGenerator(input_hidden_size=encoder_hidden_size,
                                    hidden_size=decoder_hidden_size,
                                    num_class=num_class,
                                    dropout=dropout)

    def forward(self, inputs):
        # encode logs-positions and logs
        pos_features = self.pos_encoder(inputs)
        log_features = self.encoder(pos_features, inputs)
        # predict next log entries
        outputs = self.decoder(log_features)

        return outputs
