
import os
import tempfile
from argparse import ArgumentParser

BASE_CONFIG = '\n'.join([
    'version: "1.0"',
    'tabs:',
    '  parcall:',
    f'    root: "{os.getcwd()}"',
    '    title: "parcall"',
    '    panes:',
    ''
])


def make_panel(*commands, row, col):
    panel = [
        f'    - position: "{col+1}/{row+1}"',
        '      commands:',
        '        - "reset"',
        *[f'        - "{command}"' for command in commands],
        ''
    ]
    return '\n'.join(panel)


if __name__ == '__main__':
    parser = ArgumentParser(add_help=False)
    parser.add_argument('--exp_id', '-e', type=int)
    parser.add_argument('--nrows', '-r', default=1, type=int)
    parser.add_argument('--ncols', '-c', default=1, type=int)

    args = parser.parse_args()

    if args.exp_id == 1:
        exp_runner = 'run_exp_01.sh'
    elif args.exp_id == 2:
        exp_runner = 'run_exp_02.sh'
    else:
        raise ValueError(
            f'`exp_runner` must be one of (1 | 2), is {args.exp_id}.'
        )

    remote_call = f'bash /Net/Groups/BGI/people/bkraft/git/q10hybrid/itomate/{exp_runner} 0 --new_study'
    os.system(f"ssh -t luga '{remote_call}'")

    fd, path = tempfile.mkstemp(suffix='.config')

    config = BASE_CONFIG

    i = 1
    for row in range(args.nrows):
        for col in range(args.ncols):
            remote_call_parallel =\
                f'bash /Net/Groups/BGI/people/bkraft/git/q10hybrid/itomate/{exp_runner} {col}; bash -l'
            remote_call_parallel =\
                f"ssh -t luga '{remote_call_parallel}'"
            config += make_panel(remote_call_parallel, row=row, col=col)
            i += 1

    try:
        with os.fdopen(fd, 'w') as tmp:
            # do stuff with temp file
            tmp.write(config)
        os.system(f'itomate -c {path}')
    finally:
        os.remove(path)
