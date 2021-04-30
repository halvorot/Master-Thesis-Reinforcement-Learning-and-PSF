import argparse
import hashlib
import os
import shutil
import subprocess
from pathlib import Path

ENVS = [
    'VariableWindLevel0-v17',
    'VariableWindLevel1-v17',
    'VariableWindLevel2-v17',
    'VariableWindLevel3-v17',
    'VariableWindLevel4-v17',
    'VariableWindLevel5-v17'
]

sbatch_dir = Path("sbatch")


def create_run_files():
    parser = argparse.ArgumentParser("Python implementation of spawning a lot of sbatches on train."
                                     "The env currently to run are ENVS. Queues with and without PSF")

    parser.add_argument(
        '--num_cpus',
        type=int,
        default=12,
        help='Number of cores to ask for. Default is 12 with good reason'
    )
    parser.add_argument(
        '--timesteps',
        type=int,
        default=10000000,
        help='Number of timesteps to train the agent. Default=10000000',
    )
    parser.add_argument(
        '--agent',
        help='Path to the RL agent to continue training from.',
    )
    parser.add_argument(
        '--note',
        type=str,
        default=None,
        help="Note with additional info about training"
    )
    parser.add_argument(
        '--no_reporting',
        help='Skip reporting to increase framerate',
        action='store_true'
    )
    parser.add_argument(
        '--psf_T',
        type=int,
        default=10,
        help='psf horizon'
    )
    args = parser.parse_args()

    s_config = ["#!/bin/sh\n"
                "# SBATCH -J RL_w_PSF\n"  # Sensible name for the job"
                "# SBATCH -N 1\n"  # Allocate 2 nodes for the job"
                "# SBATCH --ntasks-per-node=1\n"  # 1 task per node"
                f"# SBATCH -c {args.num_cpus}\n"
                "# SBATCH -t 24:00:00\n"  # Upper time limit for the job"
                "# SBATCH -p CPUQ\n"]

    modules = "module load fosscuda/2020b\n"

    conda_hack = "source /cluster/apps/eb/software/Anaconda3/2020.07/etc/profile.d/conda.sh\n"
    conda = "conda activate gym-rl-mpc\n"

    common_setup = s_config + modules + conda_hack + conda

    try:
        shutil.rmtree(sbatch_dir)
    except FileNotFoundError:
        pass

    os.mkdir(sbatch_dir)
    opts = ""
    for k, v in args.__dict__.items():
        opts += f" --{k} {v}" if v is not None else ''
    for psf_opt in ["", "--psf"]:
        for env in ENVS:
            train_str = f'python train.py --env {env}' + psf_opt + opts
            text = common_setup + train_str
            filename = hashlib.md5(text.encode()).hexdigest()
            with open(Path(sbatch_dir, filename + '.sh'), 'w') as f:
                f.write(text)

    return sbatch_dir


def call_sbatch(batch_loc):
    filenames = os.listdir(batch_loc)
    for f in filenames:
        print(Path(batch_loc, f).absolute())
        subprocess.run(["sbatch", str(Path(batch_loc, f))])


if __name__ == '__main__':
    loc = create_run_files()
    call_sbatch(loc)
