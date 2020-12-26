
from pytorch_lightning import seed_everything

from project.model_configs import feedforward, BaseConfig
from utils.optuna import Objective, get_study


def test_regression(fast_dev_run, resume=False, resume_version='latest'):
    config = BaseConfig(resume_policy=resume)

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
