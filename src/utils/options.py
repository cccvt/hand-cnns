import datetime
import os
import subprocess
import sys

from src.utils import filesys


def process_args(args, save_folder=None):
    # Print  options
    opts = vars(args)

    print('---- Options ----')
    for k, v in sorted(opts.items()):
        print('{option}: {value}'.format(option=k, value=v))
    print('-----------------\n')

    # Save options
    if save_folder is not None:
        filesys.mkdir(save_folder)
        opt_path = os.path.join(save_folder, 'opt.txt')
        with open(opt_path, 'a') as opt_file:
            opt_file.write('====== Options ======\n')
            for k, v in sorted(opts.items()):
                opt_file.write(
                    '{option}: {value}\n'.format(option=str(k), value=str(v)))
            git_hash = subprocess.check_output(['git', 'rev-parse', 'HEAD'])
            git_branch = subprocess.check_output(
                ['git', 'rev-parse', '--abbrev-ref', 'HEAD'])
            opt_file.write('git hash: {}\n'.format(git_hash.strip()))
            opt_file.write('git branch: {}\n'.format(git_branch.strip()))
            opt_file.write('launched {} at {}\n'.format(
                str(sys.argv[0]), str(datetime.datetime.now())))
