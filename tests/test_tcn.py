
from pytorch_lightning import seed_everything

from project.model_configs import tcn, Config
from utils.optuna_utils import Objective, get_study
from dummy_data import DataSequential


def test_tcn(fast_dev_run, resume=False, resume_version='latest'):
    config = Config(study_name='tcn_test', resume_policy=resume)

    seed_everything(config.SEED)

    study = get_study(config)

    study.optimize(
        Objective(config, tcn, wandb_offline=False, data_module=DataSequential, data_module_kwargs={'seq_last': True}),
        n_trials=12 if fast_dev_run else 20)

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
    test_tcn(False, resume=True)
