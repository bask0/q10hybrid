
import os
import shutil

log_dir = './logs/experiment_01/'
if os.path.isdir(log_dir):
    shutil.rmtree(log_dir)

fix_args = [
    # '--fast_dev_run 1'
    '--limit_train_batches 0.1',
    '--max_epochs 25',
    '--log_every_n_steps 1',
    f'--default_root_dir {log_dir}'
]

cmd = 'python train.py ' + ' '.join(fix_args)

for qinit in [0.5, 1.0, 1.5, 2.0]:
    for i in range(10):
        dynamic_args = [
            f'--q10_init {qinit}'
        ]
        cmd_i = cmd + ' ' + ' '.join(dynamic_args)
        print(f'{"-" * 80}\n RUNNING EXPERIMENT\n  >>> {cmd_i}\n{"-" * 80}')
        os.system(cmd_i)
