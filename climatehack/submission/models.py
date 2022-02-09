import torch.nn as nn
import torch
import numpy as np


class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()

        self.hid_dim = hid_dim
        self.n_layers = n_layers

        self.fc = nn.Sequential(
            nn.Linear(input_dim, emb_dim),
            nn.ReLU(),
        )

        self.rnn = nn.LSTM(
            emb_dim, hid_dim, n_layers, dropout=dropout, batch_first=True
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, src):

        # src = [batch size, 12, 128, 128]
        # this flatten pixels to be [batch_size, 12, 128*128]
        src = src.view(src.shape[0], src.shape[1], -1)
        embedded = self.fc(src)

        # embedded = [batch size, emb dim]

        outputs, (hidden, cell) = self.rnn(embedded)

        # outputs = [src len, batch size, hid dim * n directions]
        # hidden = [n layers * n directions, batch size, hid dim]
        # cell = [n layers * n directions, batch size, hid dim]

        # outputs are always from the top hidden layer

        return hidden, cell


class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()

        self.output_dim = output_dim
        self.hid_dim = hid_dim
        self.n_layers = n_layers

        self.fc_in = nn.Sequential(
            nn.Linear(output_dim, emb_dim),
            nn.ReLU(),
        )

        self.rnn = nn.LSTM(
            emb_dim, hid_dim, n_layers, dropout=dropout, batch_first=True
        )

        self.fc_out = nn.Linear(hid_dim, output_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, input, hidden, cell):

        # input = [batch size, 1, 64, 64]
        # hidden = [batch size, n layers * n directions, hid dim]
        # cell = [batch size, n layers * n directions, hid dim]

        # n directions in the decoder is 1 in this case

        # flatten images
        input = input.reshape(input.shape[0], input.shape[1], -1)
        embedded = self.fc_in(input)

        output, (hidden, cell) = self.rnn(embedded, (hidden, cell))

        # output = [batch size, hid dim * n directions]
        # hidden = [batch size, n layers * n directions, hid dim]
        # cell = [batch size, n layers * n directions, hid dim]

        prediction = self.fc_out(output)

        # prediction = [batch size, output dim]

        return prediction, hidden, cell


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.device = device

        assert (
            encoder.hid_dim == decoder.hid_dim
        ), "Hidden dimensions of encoder and decoder must be equal!"
        assert (
            encoder.n_layers == decoder.n_layers
        ), "Encoder and decoder must have equal number of layers!"

    def forward(self, srcs, trgs, teacher_forcing_ratio=0.5):

        # srcs = [batch size, 12, 128, 128]
        # trgs = [batch_size, 24, 64, 64]
        # teacher_forcing_ratio is probability to use teacher forcing
        # e.g. if teacher_forcing_ratio is 0.75 we use ground-truth inputs 75% of the time

        batch_size = trgs.shape[1]

        # tensor to store decoder outputs
        outputs = torch.zeros(trgs.shape).to(self.device)

        # last hidden state of the encoder is used as the initial hidden state of the decoder
        hidden, cell = self.encoder(srcs)

        # first input to the decoder is the last image in the hour
        # shape: (batch_size, 1, 64, 64)
        input = srcs[:, -1, 32:96, 32:96]
        input = torch.unsqueeze(input, 1)

        for t in range(24):
            output, hidden, cell = self.decoder(input, hidden, cell)
            # images of shape (batch_size, 64, 64)
            output = output.view(output.shape[0], trgs.shape[2], trgs.shape[3])
            outputs[:, t, :, :] = output

            # decide if we are going to use teacher forcing or not
            teacher_force = np.random.random() < teacher_forcing_ratio

            # if teacher forcing, use actual next token as next input
            # if not, use predicted image
            input = trgs[:, t, :, :] if teacher_force else output
            input = torch.unsqueeze(input, 1)

        return outputs
