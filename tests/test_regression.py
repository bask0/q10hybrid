
from pytorch_lightning import seed_everything

from project.model_configs import feedforward
from utils.optuna_utils import Objective, get_study
from utils.config_utils import BaseConfig


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


def test_regression(fast_dev_run, resume=False, resume_version='latest'):
    config = HConfig(resume_policy=resume)

    seed_everything(config.SEED)

    study = get_study(config)

    study.optimize(
        Objective(config, feedforward, wandb_offline=False),
        n_trials=12 if fast_dev_run else 500)

    print('Best trial:')
    trial = study.best_trial

    print('  Value: {}'.format(trial.value))

    print('  Params: ')
    for key, value in trial.params.items():
        print('    {}: {}'.format(key, value))

    # config.self_destruct_study()

    if trial.value > 0.5:
        raise AssertionError(f'model run performance above 0.7: {study.best_value}.')


if __name__ == "__main__":
    test_regression(True, resume='version_1')
