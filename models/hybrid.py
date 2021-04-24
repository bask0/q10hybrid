"""Feedforward multilayer."""

import xarray as xr
import numpy as np
from argparse import ArgumentParser
from typing import List, Dict, Tuple

import torch
import torch.nn.functional as torchf
import pytorch_lightning as pl

from utils.data_utils import Normalize

from models.feedforward import FeedForward


class Q10Model(pl.LightningModule):
    def __init__(
            self,
            features: List[str],
            targets: List[str],
            norm: Normalize,
            ds: xr.Dataset,
            q10_init: int = 1.5,
            hidden_dim: int = 128,
            num_layers: int = 2,
            dropout: float = 0.0,
            activation='relu',
            learning_rate: int = 1e-3) -> None:
        """Hybrid Q10 model.

        Note that restoring is not working currently as the model training is only taking
        some minutes.
        """

        super().__init__()
        self.save_hyperparameters(
            'features',
            'targets',
            'q10_init',
            'hidden_dim',
            'num_layers',
            'dropout',
            'activation',
            'learning_rate',
        )

        self.features = features
        self.targets = targets

        self.q10_init = q10_init

        self.input_norm = norm.get_normalization_layer(variables=self.features, invert=False, stack=True)
        self.nn = FeedForward(
            num_inputs=len(self.features),
            num_outputs=len(self.targets),
            num_hidden=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            activation=activation
        )
        self.target_norm = norm.get_normalization_layer(variables=self.targets, invert=False, stack=True)
        self.target_denorm = norm.get_normalization_layer(variables=self.targets, invert=True, stack=True)

        self.criterion = torch.nn.MSELoss()

        self.q10 = torch.nn.Parameter(torch.ones(1) * self.q10_init)
        self.ta_ref = 15.0

        # Used for strring results.
        self.ds = ds

        # Error if more than 100000 steps--ok here, but careful if you copy code for other projects!.
        self.q10_history = np.zeros(100000, dtype=np.float32) * np.nan

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:

        # Note that `x` is a dict of features and targets, input_norm extracts *only* features and stacks
        # them along last dimension.
        z = self.input_norm(x)

        # Forward pass through NN.
        z = self.nn(z)

        # No denormalization done currently.
        rb = torchf.softplus(z)

        # Physical part.
        reco = rb * self.q10 ** (0.1 * (x['ta'] - self.ta_ref))

        return reco, rb

    def criterion_normed(self, y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Calculate criterion on normalized predictions and target."""
        return self.criterion(
            self.target_norm(y_hat),
            self.target_norm(y)
        )

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        # Split batch (a dict) into actual data and the time-index returned by the dataset.
        batch, _ = batch

        # self(...) calls self.forward(...) with some extras. The `rb` is not needed here.
        reco_hat, _ = self(batch)

        # Calculate loss on normalized data.
        loss = self.criterion_normed(reco_hat, batch['reco'])

        # Save Q10 values, we want to know how they evolve with training,
        self.q10_history[self.global_step] = self.q10.item()

        # Logging.
        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        self.log('q10', self.q10, on_step=True, on_epoch=False, prog_bar=False, logger=True)
        return loss

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> Dict[str, torch.Tensor]:
        # Split batch (a dict) into actual data and the time-index returned by the dataset.
        batch, idx = batch

        # self(...) calls self.forward(...) with some extras. The `rb` is not needed here.
        reco_hat, rb_hat = self(batch)

        # Calculate loss on normalized data.
        loss = self.criterion_normed(reco_hat, batch['reco'])

        # Calculate loss on normalized data.
        self.log('valid_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        # This dict is available in `validation_epoch_end`.     
        return {'reco_hat': reco_hat, 'rb_hat': rb_hat, 'idx': idx}

    def validation_epoch_end(self, validation_step_outputs) -> None:
        # Iterate results from each validation step.
        for item in validation_step_outputs:
            reco_hat = item['reco_hat'][:, 0].cpu()
            rb_hat = item['rb_hat'][:, 0].cpu()
            idx = item['idx'].cpu()

            # Assign predictions to the right time steps.
            self.ds['reco_pred'].values[self.current_epoch, idx] = reco_hat
            self.ds['rb_pred'].values[self.current_epoch, idx] = rb_hat

    def test_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> Dict[str, torch.Tensor]:
        # Evaluation on test set.
        batch, _ = batch
        reco_hat, _ = self(batch)
        loss = self.criterion_normed(reco_hat, batch['reco'])
        self.log('test_loss', loss)

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)

    @staticmethod
    def add_model_specific_args(parent_parser: ArgumentParser) -> ArgumentParser:
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--hidden_dim', type=int, default=8)
        parser.add_argument('--num_layers', type=int, default=2)
        parser.add_argument('--q10_init', default=1.0, type=float)
        parser.add_argument('--learning_rate', type=float, default=0.001)
        return parser
