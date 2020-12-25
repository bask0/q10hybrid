
import pytorch_lightning as pl
from pytorch_lightning import Callback

import torch
from torch import nn
from torch import Tensor
from torch.optim.lr_scheduler import _LRScheduler as LRScheduler

from functools import lru_cache
from typing import Union, Iterable, Dict, Optional

from project.optim import LRStrategy


class LightningNet(pl.LightningModule):
    """Standard lightning module wrapping a PyTorch module."""
    def __init__(
            self,
            model: nn.Module,
            lr: float,
            batch_size: int,
            max_epochs: int,
            learning_strategy: str = 'adamwr',
            num_warmup_batches: Union[int, str] = 10) -> None:
        """Standard lightning module wrapping a PyTorch module.

        Note: this class should take hyperparameters regarding the training
        process. Model hyperparameters should be handled in the PyTorch module.

        Learning strategy: there are many optimizer / learning rate scheduler
        combinations that work well in practie. The choices are limited here,
        extend `src/optim:LEarningStrategy` to add more options.

        DOTO: hparam handling

        Args:
            model (nn.Module): a PyTorch model.
            lr (float): the learning rate.
            batch_size (int): the batch size.
            max_epochs (int): the number of epochs, used for learning
                rate schedulers.
            learning_strategy (str): the learning strategy, one of `AdamWWR`,
                `SGD` (see doc -> `Learnign stategy` for more details).
                Defaults to `SGD`.
            num_warmup_batches (int): number of warmup batches, the learning
                rate is scaled from 0 to the the LR ins n steps.
        """

        super().__init__()

        self.lr = lr
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.num_warmup_batches = num_warmup_batches

        self.model = model

        self.save_hyperparameters()

    def forward(self, data: Tensor) -> Tensor:
        """Model forward pass. Do not call directly.

        Args:
            data (Tensor): input data.

        Returns:
            Tensor: the model output.
        """
        return self.model(data)

    def training_step(
            self,
            batch: Iterable[Tensor],
            batch_idx: int) -> Tensor:
        """A single training step.

        Args:
            batch (Iterable[Tensor]): the bach, x, y tuble.
            batch_idx (int): the batch index (required by pl).

        Returns:
            Tensor: The batch loss.
        """
        x, y = batch
        y_hat = self(x)
        loss = nn.functional.mse_loss(y_hat, y)

        self.log('train_loss', loss)

        return loss

    def validation_step(
            self,
            batch: Iterable[Tensor],
            batch_idx: int) -> None:
        """A single validation step.

        Args:
            batch (Iterable[Tensor]): the bach, x, y tuble.
            batch_idx (int): the batch index (required by pl).

        """

        x, y = batch
        y_hat = self(x)

        loss = nn.functional.mse_loss(y_hat, y)

        self.log('val_loss', loss, on_epoch=True)

    @lru_cache()
    def get_num_batches_per_epoch(self):
        return len(self.train_dataloader())

    def configure_optimizers(
            self) -> Dict[torch.optim.Optimizer, Optional[LRScheduler]]:
        """Returns an optimizer configuration.

        Returns:
            Dict[torch.optim.Optimizer, Optional[LRScheduler]]: the optimizer,
                and optionally.
        """

        lr_strat = LRStrategy(
            lr=self.lr,
            weight_decay=0.001,
            optimizer='adamw',
            scheduler='cosine'
        )

        optimizer = lr_strat.get_optimizer(self)
        scheduler = lr_strat.get_scheduler(
            optimizer=optimizer,
            max_epochs=self.max_epochs,
            batches_per_epoch=self.get_num_batches_per_epoch()
        )

        return [optimizer], [scheduler]


class MetricsCallback(Callback):
    """PyTorch Lightning metric callback."""

    def __init__(self) -> None:
        super().__init__()
        self.metrics = []

    def on_validation_end(
            self,
            trainer: pl.Trainer,
            pl_module: pl.LightningModule):

        self.metrics.append(trainer.callback_metrics)
