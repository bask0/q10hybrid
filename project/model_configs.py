"""A collection of model configurations."""

import optuna
import pytorch_lightning as pl

from models.feedforward import FeedForward
from models.tcn import TemporalConvNet

from utils.pytorch_lighnting import LightningNet
from utils.config import BaseConfig


class HConfig(BaseConfig):
    """Global model configuration (does not change / is not tunable).
    """
    def __init__(self, study_name: str = 'study', batch_size: int = 10, *args, **kwargs):
        """Model config.

        The batch size can be set manually as different models may require different sized batches.

        Args:
            study_name (str, optional): the study name. Defaults to `study`.
            batch_size (int, optional): the batch size. Defaults to 10.
        """
        super(HConfig, self).__init__(*args, **kwargs)

        self.STUDY_NAME = study_name

        self.NUM_INPUTS = 1
        self.NUM_OUTPUTS = 1

        self.BATCH_SIZE = batch_size
        self.MIN_EPOCHS = 1
        self.MAX_EPOCHS = 5

        self.SEED = 23427

        # Not saved (bc. lowercase).
        self.log_freq = 10
        self.num_workers = 0
        self.val_loss_name = 'val_loss'


def feedforward(config: BaseConfig, trial: optuna.trial.Trial) -> pl.LightningModule:
    """Returns a tunable PyTorch lightning feedforward module.

    Args:
        config (BaseConfig): the hard-coded configuration.
        trial (optuna.Trial): optuna trial.

    Returns:
        pl.LightningModule: a lightning module.
    """

    model = FeedForward(
        num_inputs=config.NUM_INPUTS,
        num_outputs=config.NUM_OUTPUTS,
        num_hidden=trial.suggest_int('num_hidden', 1, 4),
        num_layers=trial.suggest_int('num_layers', 1, 2),
        dropout=trial.suggest_float('dropout', 0.0, 0.5),
        activation=trial.suggest_categorical('activation', ['relu', 'none'])
    )

    pl_model = LightningNet(
        model=model,
        lr=trial.suggest_loguniform('lr', 1e-3, 1e-0),
        batch_size=config.BATCH_SIZE,
        max_epochs=config.MAX_EPOCHS
    )

    return pl_model


def tcn(config: BaseConfig, trial: optuna.trial.Trial) -> pl.LightningModule:
    """Returns a tunable PyTorch lightning tcn module.

    Args:
        config (BaseConfig): the hard-coded configuration.
        trial (optuna.Trial): optuna trial.

    Returns:
        pl.LightningModule: a lightning module.
    """

    model = TemporalConvNet(
        num_inputs=config.NUM_INPUTS,
        num_hidden=trial.suggest_int('num_hidden', 1, 4),
        kernel_size=trial.suggest_int('kernel_size', 1, 4),
        num_layers=trial.suggest_int('kernel_size', 1, 2),
        dropout=trial.suggest_float('dropout', 0.1, 0.3)
    )

    pl_model = LightningNet(
        model=model,
        lr=trial.suggest_loguniform('lr', 1e-3, 1e-0),
        batch_size=config.BATCH_SIZE,
        max_epochs=config.MAX_EPOCHS
    )

    return pl_model
