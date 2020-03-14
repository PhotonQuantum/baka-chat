from copy import deepcopy
from io import BytesIO
from typing import Tuple

import torch
from comet_ml import Experiment
from torch import Tensor
from torch import nn
from torch import optim
from torch.utils.data import DataLoader

from ..common.dataset import Corpus, Dataset
from ..common.model import RNNModule


class Trainer(RNNModule):
    def __init__(self, corpus: Corpus, hparams: dict, cuda: bool = True):
        super().__init__(corpus, hparams)

        self.seq_size = hparams["seq_size"]
        self.lr = hparams["learning_rate"]
        self.epoch = hparams["max_epoch"]
        self.gradients_norm = hparams["gradients_norm"]

        self.state_h = None
        self.state_c = None
        self.criterion = None

        self.is_cuda = cuda
        self.best_model = None

    def prepare_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.lr)

    @staticmethod
    def prepare_criterion():
        return nn.CrossEntropyLoss()

    def prepare_dataset(self) -> DataLoader:
        return DataLoader(Dataset(self.corpus, self.seq_size), batch_size=self.batch_size,
                          pin_memory=True, drop_last=True)

    def before_train_epoch(self):
        self.state_h, self.state_c = self.zero_state()  # reset hidden layers
        if self.is_cuda:
            self.state_h = self.state_h.cuda()
            self.state_c = self.state_c.cuda()

    def train_step(self, batch: Tuple[Tensor, Tensor]):
        x, y = batch
        if self.is_cuda:
            x, y = x.cuda(), y.cuda()

        # forward and calculate loss
        logits, (self.state_h, self.state_c) = self.forward(x, (self.state_h, self.state_c))
        loss = self.criterion(logits.transpose(1, 2), y)

        # detach hidden layers
        self.state_h, self.state_c = self.state_h.detach(), self.state_c.detach()

        return loss

    def fit(self, experiment: Experiment, cuda: bool = True) -> dict:
        # Prepare comet.ml logger
        experiment.log_parameters(self.hparams)  # log hyper parameters

        # noinspection PyUnresolvedReferences
        torch.backends.cudnn.benchmark = True  # enable cudnn benchmark mode for better performance
        self.is_cuda = cuda
        if self.is_cuda:
            self.cuda()  # move model to cuda

        train_set = self.prepare_dataset()
        self.criterion = self.prepare_criterion()
        optimizer = self.prepare_optimizers()

        best_model = {"loss": 999, "epoch": -1, "model": None}

        self.train()  # set the model to train mode

        with experiment.train():
            for e in range(self.epoch):
                self.before_train_epoch()

                batch_loss = 0
                for batch_idx, batch in enumerate(train_set):
                    optimizer.zero_grad()

                    loss = self.train_step(batch)

                    batch_loss += loss.item()

                    loss.backward()

                    _ = torch.nn.utils.clip_grad_norm_(self.parameters(), self.gradients_norm)

                    optimizer.step()

                avg_loss = batch_loss / len(train_set)
                experiment.log_metric("epoch_loss", avg_loss, step=e)
                # noinspection PyUnboundLocalVariable
                if avg_loss < best_model["loss"]:
                    best_model["loss"] = avg_loss
                    best_model["epoch"] = e
                    best_model["state_dict"] = deepcopy(self.state_dict())  # detach all weights with deepcopy

        return best_model

    @staticmethod
    def dumps(state_dict):
        tmp_buffer = BytesIO()
        torch.save(state_dict, tmp_buffer)
        tmp_buffer.seek(0)
        return tmp_buffer.read()
