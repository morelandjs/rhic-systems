"""
Computes model observables to match experimental data.
Prints all model data when run as a script.

Model data files are expected with the file structure
:file:`model_output/{design}/{system}/{design_point}.dat`, where
:file:`{design}` is a design type, :file:`{system}` is a system string, and
:file:`{design_point}` is a design point name.

For example, the structure of my :file:`model_output` directory is ::

    model_output
    ├── main
    │   ├── PbPb2760
    │   │   ├── 000.dat
    │   │   └── 001.dat
    │   └── PbPb5020
    │       ├── 000.dat
    │       └── 001.dat
    └── validation
        ├── PbPb2760
        │   ├── 000.dat
        │   └── 001.dat
        └── PbPb5020
            ├── 000.dat
            └── 001.dat

I have two design types (main and validation), two systems, and my design
points are numbered 000-499 (most numbers omitted for brevity).

Data files are expected to have the binary format created by my `heavy-ion
collision event generator
<https://github.com/jbernhard/heavy-ion-collisions-osg>`_.

Of course, if you have a different data organization scheme and/or format,
that's fine.  Modify the code for your needs.
"""

import logging
from pathlib import Path
import pickle

import numpy as np
from sklearn.externals import joblib

from . import workdir, cachedir, systems, lazydict, expt
from .design import Design


class ModelData:
    """
    Helper class for event-by-event model data.  Reads binary data files and
    computes centrality-binned observables.

    """
    def __init__(self, *files, scale_norm=1):
        # read each file using the above dtype and treat each as a minimum-bias
        # event sample
        def load_events(f):
            logging.debug('loading %s', f)
            d = np.load(f)
            d = np.array(d['mult'], dtype=[('mult', '<f8')])
            d.sort(order='mult')
            return d

        self.scale_norm = scale_norm

        self.events = [load_events(f) for f in files]

    def observables_like(self, data, *keys):
        """
        Compute the same centrality-binned observables as contained in `data`
        with the same nested dict structure.

        This function calls itself recursively, each time prepending to `keys`.

        """
        try:
            x = data['x']
            cent = data['cent']
        except KeyError:
            return {
                k: self.observables_like(v, k, *keys)
                for k, v in data.items()
            }

        def _compute_bin():
            """
            Choose a function to compute the current observable for a single
            centrality bin.

            """
            obs_stack = list(keys)
            obs = obs_stack.pop()

            if obs in ['dNch_deta', 'dET_deta']:
                return lambda events: self.scale_norm * events['mult'].mean()

        compute_bin = _compute_bin()

        def compute_all_bins(events):
            n = events.size
            bins = [
                events[int((1 - b/100)*n):int((1 - a/100)*n)]
                for a, b in cent
            ]

            return list(map(compute_bin, bins))

        return dict(
            x=x, cent=cent,
            Y=np.array(list(map(compute_all_bins, self.events))).squeeze()
        )


def _data(system, dataset='main'):
    """
    Compute model observables for the given system and dataset.

    dataset may be one of:

        - 'main' (training design)
        - 'validation' (validation design)
        - 'map' (maximum a posteriori, i.e. "best-fit" point)

    """
    if dataset not in {'main', 'validation', 'map'}:
        raise ValueError('invalid dataset: {}'.format(dataset))

    files = (
        [Path(workdir, 'model_output', dataset, '{}.dat'.format(system))]
        if dataset == 'map' else
        [
            Path(workdir, 'model_output', dataset, system, '{}.npz'.format(p))
            for p in
            Design(system, validation=(dataset == 'validation')).points
        ]
    )

    cachefile = Path(cachedir, 'model', dataset, '{}.pkl'.format(system))

    if cachefile.exists():
        # use the cache unless any of the model data files are newer
        # this DOES NOT check any other logical dependencies, e.g. the
        # experimental data
        # to force recomputation, delete the cache file
        mtime = cachefile.stat().st_mtime
        if all(f.stat().st_mtime < mtime for f in files):
            logging.debug('loading observables cache file %s', cachefile)
            return joblib.load(cachefile)
        else:
            logging.debug('cache file %s is older than event data', cachefile)
    else:
        logging.debug('cache file %s does not exist', cachefile)

    logging.info(
        'loading %s/%s data and computing observables',
        system, dataset
    )

    data = expt.data[system]
    data = ModelData(*files, scale_norm=1).observables_like(data)

    logging.info('writing cache file %s', cachefile)
    cachefile.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(data, cachefile, protocol=pickle.HIGHEST_PROTOCOL)

    return data


data = lazydict(_data, 'main')
validation_data = lazydict(_data, 'validation')
map_data = lazydict(_data, 'map')


if __name__ == '__main__':
    from pprint import pprint
    for s in systems:
        d = data[s]
        print(s)
        pprint(d)
