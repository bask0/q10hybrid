import optuna
import pytorch_lightning as pl

from project.layer_feedforward import FeedForward
from project.pl_utils import LightningNet
from project.utils import BaseConfig


def feedforward(config: BaseConfig, trial: optuna.trial.Trial) -> pl.LightningModule:
    """Returns a tunable PyTorch lightning module.

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
        dropout=trial.suggest_float('dropout', 0.1, 0.3),
        activation=trial.suggest_categorical('activation', ['relu', 'none'])
    )

    pl_model = LightningNet(
        model=model,
        lr=trial.suggest_loguniform('lr', 1e-3, 1e-0),
        batch_size=config.BATCH_SIZE,
        max_epochs=config.MAX_EPOCHS
    )

    return pl_model
