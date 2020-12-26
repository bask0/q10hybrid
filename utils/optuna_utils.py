
import optuna
from optuna.integration import PyTorchLightningPruningCallback

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import LearningRateMonitor

from typing import Callable, Dict, Optional
from numbers import Number

from project.utils import BaseConfig
from project.pl_utils import MetricsCallback


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
