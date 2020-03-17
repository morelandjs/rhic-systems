"""
Runs the TRENTO initial condition model at each of
the design points, and saves the array of event attributes.

"""
import logging
import multiprocessing
import numpy as np
from pathlib import Path
import subprocess

from . import workdir


def run_cmd(*args):
    """
    Run and log a subprocess.

    """
    cmd = ' '.join(args)
    logging.info('running command: %s', cmd)

    try:
        proc = subprocess.run(
            cmd.split(), check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True
        )
    except subprocess.CalledProcessError as e:
        logging.error(
            'command failed with status %d:\n%s',
            e.returncode, e.output.strip('\n')
        )
        raise
    else:
        logging.debug(
            'command completed successfully:\n%s',
            proc.stdout.strip('\n')
        )
        return proc


def run_trento(config_file):

    # model output path
    outfile = Path(str(config_file).replace('design', 'model_output'))
    outfile.parent.mkdir(parents=True, exist_ok=True)

    # run trento using the config file
    proc = run_cmd(f'trento -c {config_file}')
    mult = np.loadtxt(proc.stdout.splitlines(), usecols=3)
    np.savez_compressed(outfile, mult=mult)


if __name__ == '__main__':

    # trento design point and model output directories
    modeldir = workdir / 'model_output'
    designdir = workdir / 'design'
    modeldir.mkdir(exist_ok=True)

    # number of available cpu cores
    ncpu = multiprocessing.cpu_count()

    # run the model at each of the design points
    #for design_type in ['main', 'validation']:
    for design_type in ['main', ]:
        multiprocessing.Pool(ncpu).map(
            run_trento, (workdir / 'design' / design_type).glob('*/*')
        )
