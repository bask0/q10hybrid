
import optuna
from optuna.integration import PyTorchLightningPruningCallback

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import LearningRateMonitor

import os
import re
from typing import Callable, Dict, Optional
from numbers import Number

from utils.config_utils import BaseConfig
from utils.pl_utils import MetricsCallback


class Objective(object):
    def __init__(
            self,
            config: BaseConfig,
            model_generator: Callable,
            data_module: pl.LightningDataModule,
            resume: bool = False,
            fast_dev_run: bool = False,
            wandb_offline: bool = False,
            data_module_kwargs: Optional[Dict] = {}) -> None:
        """Defines an optuna `objective`.

        Args:
            config (BaseConfig): the hardcoded configuration.
            model_generator (Callable): a function that takes a `config` and a `trail` and returns
                a pl.LightningModule.
            data_module (pl.LightningDataModule): a pytorch-lightning data module, must take at least an
                argument `batch_size`. Further arguments can be passed via `data_module_kwargs`.
            resume (bool): whether to resume a previous run.
            fast_dev_run (bool, optional): wheter to run a fast dev run. Defaults to `False`.
            wandb_offline (bool, optional): run offline. If `True`, do not transfer to wandb server.
                Defaults to `False`.
            data_module_kwargs (Optional[Dict]): Keyword arguments passed to `data_module`.
        """
        self.config = config
        self.model_generator = model_generator
        self.fast_dev_run = fast_dev_run
        self.wandb_offline = wandb_offline
        self.data_module = data_module
        self.data_module_kwargs = data_module_kwargs

    def __call__(self, trial: optuna.Trial) -> Number:
        config = self.config.copy()
        config.set_trial_name(trial)
        config.makedirs()
        config.set_trial_attrs(trial)

        checkpoint_callback = pl.callbacks.ModelCheckpoint(
            dirpath=config.TRIAL_DIR,
            monitor=config.val_loss_name,
            save_last=True,
            save_top_k=1,
            mode='min',
            verbose=False
        )

        logger = WandbLogger(
            name=config.TRIAL_UID,
            save_dir=config.STUDY_DIR_VERSION,
            project=config.STUDY_NAME,
            tags=[config.VERSION],
            offline=self.wandb_offline,
            version=config.TRIAL_UID
        )

        resume = config.get_latest_checkpoint()

        pl_model = self.model_generator(config, trial)

        metrics_callback = MetricsCallback()
        trainer = pl.Trainer(
            min_epochs=config.MIN_EPOCHS,
            max_epochs=config.MAX_EPOCHS,
            logger=logger,
            callbacks=[
                metrics_callback,
                checkpoint_callback,
                PyTorchLightningPruningCallback(
                    trial,
                    monitor=config.val_loss_name),
                LearningRateMonitor(
                    logging_interval='step')],
            fast_dev_run=self.fast_dev_run,
            log_every_n_steps=config.log_freq,
            resume_from_checkpoint=resume
        )

        logger.watch(pl_model, log='gradients', log_freq=config.log_freq)

        params = {
            **config.get_params(),
            **trial.params
        }

        logger.log_hyperparams(params)

        trainer.fit(pl_model, self.data_module(config.BATCH_SIZE, **self.data_module_kwargs))

        logger.experiment.finish()

        return metrics_callback.metrics[-1][config.val_loss_name].item()


def get_study(config: BaseConfig, allow_resume=False):
    """Returns an optuna `study`.

    Args:
        config (BaseConfig): the hardcoded configuration.

    Returns:
        optuna.Study: an optuna study.
    """

    pruner = optuna.pruners.HyperbandPruner(
        min_resource=config.MIN_EPOCHS,
        max_resource=config.MAX_EPOCHS,
        reduction_factor=3)

    sampler = optuna.samplers.TPESampler(
        multivariate=True,
        seed=config.SEED
    )

    study = optuna.create_study(
        study_name=config.STUDY_NAME,
        storage=config.OPTUNA_DB,
        direction='minimize',
        pruner=pruner,
        load_if_exists=True)

    return study


def get_best_trial_dir(study: optuna.Study) -> str:
    """Get best trial dir from a study.

    Args:
        study (optuna.Study): an optuna study.

    Returns:
        str: the best trial dir.
    """
    best_trial_dir = os.path.dirname(study.best_trial.user_attrs['LAST_CKPT_PATH'])
    return best_trial_dir


def get_best_trial_checkpoint(study: optuna.Study) -> str:
    """Get checkpoint of best trial.

    Note:
        The function scans for the pattern `epoch=[int].ckpt` (pytorch lightning default) and takes the most recent 
        one, i.e., the highest integer matching the pattern. This only works if the pytorch lightning Trainer saves
        at least the 1 top (best) checkpoints (`pl.Trainer(..., save_top_k=1)`).

    Args:
        study (optuna.Study): an optuna study.

    Raises:
        ValueError: if directory is empty.
        ValueError: if no matching checkpoint pattern.
        AssertionError: if found pattern does not match any file (shouldn't happen).

    Returns:
        str: the best checkpoint path.
    """
    best_trial_dir = get_best_trial_dir(study)
    files = os.listdir(best_trial_dir)

    message = f'attempt to load best checkpoint from {"best"} failed. '

    if len(files) == 0:
        raise ValueError(
            message +
            'Directory is empty.'
        )

    epoch_nr = []
    for f in files:
        m = re.match(r'epoch=(\d+).ckpt', f)
        if m is not None:
            epoch_nr.append(int(m.group(1)))

    if len(epoch_nr) == 0:
        raise ValueError(
            message +
            'No checkpoints matching the pattern `epoch=[d].ckpt` found.'
        )

    best_epoch = f'epoch={max(epoch_nr)}.ckpt'

    if best_epoch not in files:
        raise AssertionError(
            message +
            f'no file matching the found best epoch `{best_epoch}`. Check source code.'
        )

    return os.path.join(best_trial_dir, best_epoch)