"""A collection of model configurations."""

import optuna
import pytorch_lightning as pl

from models.feedforward import FeedForward
from models.tcn import TemporalConvNet

from utils.pytorch_lighnting import LightningNet
from utils.config import BaseConfig


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
