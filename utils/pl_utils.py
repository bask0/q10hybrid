
import pytorch_lightning as pl
from pytorch_lightning import Callback

import torch
from torch import nn
from torch import Tensor
from torch.optim.lr_scheduler import _LRScheduler as LRScheduler

from functools import lru_cache
from typing import Union, Iterable, Dict, Optional

from utils.optim_utils import LRStrategy


def get_training_config(
        lr: float,
        weight_decay: float,
        max_epochs: int,
        optimizer: str = 'adamw',
        scheduler: str = 'cosine',
        num_warmup_batches: Union[int, str] = 'auto') -> Dict:
    """Returns a training configuration that can be used as kwargs to `LightningNet`.

    Args:
        lr (float): the learning rate, > 0.0.
        weight_decay (float): weight decay (L2 regulatizatiuon), > 0.0.
        max_epochs (int): the number of epochs, used for learning rate schedulers.
        optimizer (str): optimizer, one of `adamw`, `sgd` (see doc -> `Learninng stategy`
            for more details). Defaults to `sgd`.
        scheduler (str): learning rate scheduler, one of:
            * `cosine`: CosineAnnealingWithWarmup
            * `reduceonplateau`: ReduceLROnPlateauWithWarmup (currently not working)
            * `cyclic`: CyclicLR scheduler
            * `cyclic2`: CyclicLR scheduler
            * `onecycle` OneCycleLR scheduler
        num_warmup_batches (Union[int, str], optional): the number of warmup steps. Does not apply to all
            schedulers (cyclic and onecycle do start at low lr anyway). No warmup is done if `0`, one full
            epoch (gradually increasing per batch) if `auto`. Defaults to `auto`.

    Returns:
        Dict: [description]
    """

    config = dict(
        lr=lr,
        weight_decay=weight_decay,
        max_epochs=max_epochs,
        optimizer=optimizer,
        scheduler=scheduler,
        num_warmup_batches=num_warmup_batches
    )

    return config


class LightningNet(pl.LightningModule):
    """Standard lightning module wrapping a PyTorch module."""
    def __init__(
            self,
            lr: float,
            weight_decay: float,
            max_epochs: int,
            optimizer='adamw',
            scheduler='cosine',
            num_warmup_batches: Union[int, str] = 'auto') -> None:
        """Standard lightning module, should be subclassed.

        Note:
            * This class should take hyperparameters regarding the training process. Model hyperparameters should be
                handled in the PyTorch module.
            * call 'self.save_hyperparameters()' at the end of subclass `__init__()`.

        Learning strategy: there are many optimizer / learning rate scheduler
        combinations that work well in practie. The choices are limited here,
        extend `src/optim:LEarningStrategy` to add more options.

        DOTO: hparam handling

        Args:
            lr (float): the learning rate, > 0.0.
            weight_decay (float): weight decay (L2 regulatizatiuon), > 0.0.
            max_epochs (int): the number of epochs, used for learning rate schedulers.
            optimizer (str): optimizer, one of `adamw`, `sgd` (see doc -> `Learninng stategy`
                for more details). Defaults to `sgd`.
            scheduler (str): learning rate scheduler, one of:
                * `cosine`: CosineAnnealingWithWarmup
                * `reduceonplateau`: ReduceLROnPlateauWithWarmup (currently not working)
                * `cyclic`: CyclicLR scheduler
                * `cyclic2`: CyclicLR scheduler
                * `onecycle` OneCycleLR scheduler
            num_warmup_batches (Union[int, str], optional): the number of warmup steps. Does not apply to all
                schedulers (cyclic and onecycle do start at low lr anyway). No warmup is done if `0`, one full
                epoch (gradually increasing per batch) if `auto`. Defaults to `auto`.
        """

        super().__init__()

        self.lr = lr
        self.weight_decay = weight_decay
        self.max_epochs = max_epochs
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.num_warmup_batches = num_warmup_batches

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
            weight_decay=self.weight_decay,
            optimizer=self.optimizer,
            scheduler=self.scheduler
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
