from typing import Tuple

import torch
from torch import Tensor
from torch import nn

from .dataset import Corpus


class RNNModule(nn.Module):
    def __init__(self, corpus: Corpus, hparams: dict):
        super().__init__()

        self.corpus = corpus
        self.hparams = hparams
        self.lstm_size = hparams["lstm_size"]
        self.embedding_size = hparams["embed_size"]
        self.num_layers = hparams["lstm_layers"]
        self.batch_size = hparams["batch_size"]
        self.n_vocab = len(self.corpus.table_int_to_vocab)

        self.embedding = nn.Embedding(self.n_vocab, self.embedding_size)
        self.lstm = nn.LSTM(self.embedding_size,
                            self.lstm_size,
                            self.num_layers,
                            batch_first=True)
        self.dense = nn.Linear(self.lstm_size, self.n_vocab)

    def forward(self, x: Tensor, prev_state: Tuple[Tensor, Tensor]) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
        embed = self.embedding(x)
        output, state = self.lstm(embed, prev_state)
        logits = self.dense(output)

        return logits, state

    def zero_state(self) -> Tuple[Tensor, Tensor]:
        return (torch.zeros(self.num_layers, self.batch_size, self.lstm_size),
                torch.zeros(self.num_layers, self.batch_size, self.lstm_size))
