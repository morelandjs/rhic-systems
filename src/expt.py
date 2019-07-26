"""
Downloads, processes, and stores experimental data.
Prints all data when run as a script.
"""

from collections import defaultdict
import logging
import pickle
from urllib.request import urlopen

import numpy as np
import yaml

from . import cachedir, systems, parse_system


class HEPData:
    """
    Interface to a `HEPData <https://hepdata.net>`_ YAML data table.

    Downloads and caches the dataset specified by the INSPIRE record and table
    number.  The web UI for `inspire_rec` may be found at
    :file:`https://hepdata.net/record/ins{inspire_rec}`.

    If `reverse` is true, reverse the order of the data table (useful for
    tables that are given as a function of Npart).

    .. note::

        Datasets are assumed to be a function of centrality.  Other kinds of
        datasets will require code modifications.

    """
    def __init__(self, inspire_rec, table, reverse=False):
        cachefile = (
            cachedir / 'hepdata' /
            'ins{}_table{}.pkl'.format(inspire_rec, table)
        )
        name = 'record {} table {}'.format(inspire_rec, table)

        if cachefile.exists():
            logging.debug('loading from hepdata cache: %s', name)
            with cachefile.open('rb') as f:
                self._data = pickle.load(f)
        else:
            logging.debug('downloading from hepdata.net: %s', name)
            cachefile.parent.mkdir(exist_ok=True)
            with cachefile.open('wb') as f, urlopen(
                    'https://hepdata.net/download/table/'
                    'ins{}/Table{}/yaml'.format(inspire_rec, table)
            ) as u:
                self._data = yaml.load(u)
                pickle.dump(self._data, f, protocol=pickle.HIGHEST_PROTOCOL)

        if reverse:
            for v in self._data.values():
                for d in v:
                    d['values'].reverse()

    def x(self, name, case=True):
        """
        Get an independent variable ("x" data) with the given name.

        If `case` is false, perform case-insensitive matching for the name.

        """
        trans = (lambda x: x) if case else (lambda x: x.casefold())
        name = trans(name)

        for x in self._data['independent_variables']:
            if trans(x['header']['name']) == name:
                return x['values']

        raise LookupError("no x data with name '{}'".format(name))

    @property
    def cent(self):
        """
        The centrality bins as a list of (low, high) tuples.

        """
        try:
            return self._cent
        except AttributeError:
            pass

        x = self.x('centrality', case=False)

        if x is None:
            raise LookupError('no centrality data')

        try:
            cent = [(v['low'], v['high']) for v in x]
        except KeyError:
            # try to guess bins from midpoints
            mids = [v['value'] for v in x]
            width = set(a - b for a, b in zip(mids[1:], mids[:-1]))
            if len(width) > 1:
                raise RuntimeError('variable bin widths')
            d = width.pop() / 2
            cent = [(m - d, m + d) for m in mids]

        self._cent = cent

        return cent

    @cent.setter
    def cent(self, value):
        """
        Manually set centrality bins.

        """
        self._cent = value

    def y(self, name=None, **quals):
        """
        Get a dependent variable ("y" data) with the given name and qualifiers.

        """
        for y in self._data['dependent_variables']:
            if name is None or y['header']['name'].startswith(name):
                y_quals = {q['name']: q['value'] for q in y['qualifiers']}
                if all(y_quals[k] == v for k, v in quals.items()):
                    return y['values']

        raise LookupError(
            "no y data with name '{}' and qualifiers '{}'"
            .format(name, quals)
        )

    def dataset(self, name=None, maxcent=100, ignore_bins=[], **quals):
        """
        Return a dict containing:

        - **cent:** list of centrality bins
        - **x:** numpy array of centrality bin midpoints
        - **y:** numpy array of y values
        - **yerr:** subdict of numpy arrays of y errors

        `name` and `quals` are passed to `HEPData.y()`.

        Missing y values are skipped.

        Centrality bins whose upper edge is greater than `maxcent` are skipped.

        Centrality bins in `ignore_bins` [a list of (low, high) tuples] are
        skipped.

        """
        cent = []
        y = []
        yerr = defaultdict(list)

        for c, v in zip(self.cent, self.y(name, **quals)):
            # skip missing values
            # skip bins whose upper edge is greater than maxcent
            # skip explicitly ignored bins
            if v['value'] == '-' or c[1] > maxcent or c in ignore_bins:
                continue

            cent.append(c)
            y.append(v['value'])

            for err in v['errors']:
                try:
                    e = err['symerror']
                except KeyError:
                    e = err['asymerror']
                    if abs(e['plus']) != abs(e['minus']):
                        raise RuntimeError(
                            'asymmetric errors are not implemented'
                        )
                    e = abs(e['plus'])

                yerr[err.get('label', 'sum')].append(e)

        return dict(
            cent=cent,
            x=np.array([(a + b)/2 for a, b in cent]),
            y=np.array(y),
            yerr={k: np.array(v) for k, v in yerr.items()},
        )


def _data():
    """
    Curate the experimental data using the `HEPData` class and return a nested
    dict with levels

    - system
    - observable
    - subobservable
    - dataset (created by :meth:`HEPData.dataset`)

    For example, ``data['PbPb2760']['dN_dy']['pion']`` retrieves the dataset
    for pion dN/dy in Pb+Pb collisions at 2.76 TeV.

    Some observables, such as charged-particle multiplicity, don't have a
    natural subobservable, in which case the subobservable is set to `None`.

    The best way to understand the nested dict structure is to explore the
    object in an interactive Python session.

    """
    data = {s: {} for s in systems}

    # dAu200 transverse energy
    # https://arxiv.org/abs/1509.06727
    data['dAu200']['dET_deta'] = {}

    cent, x, y, yerr = zip(*[
        (( 0,  5),  2.5, 20.3, 1.7),
        (( 5, 10),  7.5, 17.4, 1.5),
        ((10, 20),   15, 15.4, 1.3),
        ((20, 30),   25, 13.2, 1.1),
        ((30, 40),   35, 11.3, 0.9),
        ((40, 50),   45,  9.5, 0.8),
        ((50, 60),   55,  7.8, 0.7),
        ((60, 70),   65,  6.3, 0.5),
    ])

    data['dAu200']['dET_deta'][None] = dict(
        cent=cent,
        x=np.array(x),
        y=np.array(y),
        yerr={'stat': np.zeros_like(yerr), 'sys': np.array(yerr)}
    )

    # He3Au200 transverse energy
    # https://arxiv.org/abs/1509.06727
    data['He3Au200']['dET_deta'] = {}

    cent, x, y, yerr = zip(*[
        (( 0,  5),  2.5, 26.7, 1.8),
        (( 5, 10),  7.5, 23.2, 1.5),
        ((10, 20),   15, 20.6, 1.4),
        ((20, 30),   25, 17.7, 1.2),
        ((30, 40),   35, 14.9, 1.0),
        ((40, 50),   45, 12.0, 0.8),
        ((50, 60),   55,  9.3, 0.6),
        ((60, 70),   65,  7.0, 0.5),
    ])

    data['He3Au200']['dET_deta'][None] = dict(
        cent=cent,
        x=np.array(x),
        y=np.array(y),
        yerr={'stat': np.zeros_like(yerr), 'sys': np.array(yerr)}
    )

    # CuCu200 transverse energy
    # https://arxiv.org/abs/1509.06727
    data['CuCu200']['dET_deta'] = {}

    cent, x, y, yerr = zip(*[
        (( 0,  5),  2.5, 166.8, 13.2),
        (( 5, 10),  7.5, 139.9, 11.1),
        ((10, 15), 12.5, 117.1,  9.3),
        ((15, 20), 17.5,  97.9,  7.8),
        ((20, 25), 22.5,  81.6,  6.5),
        ((25, 30), 27.5,  67.8,  5.4),
        ((30, 35), 32.5,  56.1,  4.4),
        ((35, 40), 37.5,  46.0,  3.6),
        ((40, 45), 42.5,  37.5,  3.0),
    ])

    data['CuCu200']['dET_deta'][None] = dict(
        cent=cent,
        x=np.array(x),
        y=np.array(y),
        yerr={'stat': np.zeros_like(yerr), 'sys': np.array(yerr)}
    )

    # CuAu200 transverse energy
    # https://arxiv.org/abs/1509.06727
    data['CuAu200']['dET_deta'] = {}

    cent, x, y, yerr = zip(*[
        (( 0,  5),  2.5, 288.3, 17.3),
        (( 5, 10),  7.5, 249.8, 15.0),
        ((10, 15), 12.5, 212.8, 12.8),
        ((15, 20), 17.5, 179.4, 10.8),
        ((20, 25), 22.5, 150.0,  9.0),
        ((25, 30), 27.5, 124.5,  7.5),
        ((30, 35), 32.5, 102.3,  6.1),
        ((35, 40), 37.5,  83.3,  5.0),
        ((40, 45), 42.5,  67.0,  4.0),
        ((45, 50), 47.5,  53.1,  3.2),
        ((50, 55), 52.5,  41.4,  2.5),
        ((55, 60), 57.5,  31.9,  1.9),
    ])

    data['CuAu200']['dET_deta'][None] = dict(
        cent=cent,
        x=np.array(x),
        y=np.array(y),
        yerr={'stat': np.zeros_like(yerr), 'sys': np.array(yerr)}
    )

    # AuAu200 transverse energy
    # https://arxiv.org/abs/1509.06727
    data['AuAu200']['dET_deta'] = {}

    cent, x, y, yerr = zip(*[
        (( 0,  5),  2.5, 599.0,  34.7),
        (( 5, 10),  7.5, 498.7,  28.9),
        ((10, 15), 12.5, 403.0,  25.0),
        ((15, 20), 17.5, 332.5,  21.2),
        ((20, 25), 22.5, 273.6,  18.6),
        ((25, 30), 27.5, 223.4,  16.4),
        ((30, 35), 32.5, 180.8,  14.3),
        ((35, 40), 37.5, 144.5,  12.6),
        ((40, 45), 42.5, 113.9,  10.9),
        ((45, 50), 47.5,  88.3,   9.3),
        ((50, 55), 52.5,  67.1,   8.1),
        ((55, 60), 57.5,  50.0,   6.7),
    ])

    data['AuAu200']['dET_deta'][None] = dict(
        cent=cent,
        x=np.array(x),
        y=np.array(y),
        yerr={'stat': np.zeros_like(yerr), 'sys': np.array(yerr)}
    )

    # UU193 transverse energy
    # https://arxiv.org/abs/1509.06727
    data['UU193']['dET_deta'] = {}

    cent, x, y, yerr = zip(*[
        (( 0,  5),  2.5, 783.0,  46.1),
        (( 5, 10),  7.5, 625.6,  36.9),
        ((10, 15), 12.5, 504.0,  29.7),
        ((15, 20), 17.5, 406.2,  23.9),
        ((20, 25), 22.5, 325.9,  19.2),
        ((25, 30), 27.5, 259.2,  15.3),
        ((30, 35), 32.5, 203.7,  12.0),
        ((35, 40), 37.5, 157.8,   9.3),
        ((40, 45), 42.5, 119.9,   7.1),
        ((45, 50), 47.5, 89.16,   5.3),
    ])

    data['UU193']['dET_deta'][None] = dict(
        cent=cent,
        x=np.array(x),
        y=np.array(y),
        yerr={'stat': np.zeros_like(yerr), 'sys': np.array(yerr)}
    )

    # PbPb2760 transverse energy
    # ignore bin 0-5 since it's redundant with 0-2.5 and 2.5-5
    #dset = HEPData(1427723, 1).dataset('$E_{T}$', ignore_bins=[(0, 5)])
    #dset['yerr']['sys'] = dset['yerr'].pop('sys,total')
    #data['PbPb2760']['dET_deta'] = {None: dset}

    # PbPb5020 dNch/deta
    #data['PbPb5020']['dNch_deta'] = {
    #    None: HEPData(1410589, 2).dataset(
    #        r'$\mathrm{d}N_\mathrm{ch}/\mathrm{d}\eta$')}

    # XeXe5440 dNch/deta
    # https://arxiv.org/abs/1805.04432
    #data['XeXe5440']['dNch_deta'] = {}

    #cent, x, y, yerr = zip(*[
    #    ((0.0, 2.5), 1.25, 1238,  25),
    #    ((2.5, 5.0), 3.75, 1096,  27),
    #    ((5.0, 7.5), 6.25,  986,  25),
    #    ((7.5,  10), 8.75,  891,  24),
    #    (( 10,  20),   15,  706,  17),
    #    (( 20,  30),   25,  478,  11),
    #    (( 30,  40),   35,  315,   8),
    #    (( 40,  50),   45,  198,   5),
    #    (( 50,  60),   55,  118,   3),
    #    (( 60,  70),   65, 64.7,   2),
    #    (( 70,  80),   75, 32.0, 1.3),
    #    (( 80,  90),   85, 13.3, 0.9),
    #])

    #data['XeXe5440']['dNch_deta'][None] = dict(
    #    cent=cent,
    #    x=np.array(x),
    #    y=np.array(y),
    #    yerr={'stat': np.zeros_like(yerr), 'sys': np.array(yerr)}
    #)

    return data


#: A nested dict containing all the experimental data, created by the
#: :func:`_data` function.
data = _data()


def cov(
        system, obs1, subobs1, obs2, subobs2,
        stat_frac=1e-4, sys_corr_length=100, cross_factor=.8,
        corr_obs={
            frozenset({'dNch_deta', 'dET_deta', 'dN_dy'}),
        }
):
    """
    Estimate a covariance matrix for the given system and pair of observables,
    e.g.:

    >>> cov('PbPb2760', 'dN_dy', 'pion', 'dN_dy', 'pion')
    >>> cov('PbPb5020', 'dN_dy', 'pion', 'dNch_deta', None)

    For each dataset, stat and sys errors are used if available.  If only
    "summed" error is available, it is treated as sys error, and `stat_frac`
    sets the fractional stat error.

    Systematic errors are assumed to have a Gaussian correlation as a function
    of centrality percentage, with correlation length set by `sys_corr_length`.

    If obs{1,2} are the same but subobs{1,2} are different, the sys error
    correlation is reduced by `cross_factor`.

    If obs{1,2} are different and uncorrelated, the covariance is zero.  If
    they are correlated, the sys error correlation is reduced by
    `cross_factor`.  Two different obs are considered correlated if they are
    both a member of one of the groups in `corr_obs` (the groups must be
    set-like objects).  By default {Nch, ET, dN/dy} are considered correlated
    since they are all related to particle / energy production.

    """
    def unpack(obs, subobs):
        dset = data[system][obs][subobs]
        yerr = dset['yerr']

        try:
            stat = yerr['stat']
            sys = yerr['sys']
        except KeyError:
            stat = dset['y'] * stat_frac
            sys = yerr['sum']

        return dset['x'], stat, sys

    x1, stat1, sys1 = unpack(obs1, subobs1)
    x2, stat2, sys2 = unpack(obs2, subobs2)

    if obs1 == obs2:
        same_obs = (subobs1 == subobs2)
    else:
        # check if obs are both in a correlated group
        if any({obs1, obs2} <= c for c in corr_obs):
            same_obs = False
        else:
            return np.zeros((x1.size, x2.size))

    # compute the sys error covariance
    C = (
        np.exp(-.5*(np.subtract.outer(x1, x2)/sys_corr_length)**2) *
        np.outer(sys1, sys2)
    )

    if same_obs:
        # add stat error to diagonal
        C.flat[::C.shape[0]+1] += stat1**2
    else:
        # reduce correlation for different observables
        C *= cross_factor

    return C


def print_data(d, indent=0):
    """
    Pretty print the nested data dict.

    """
    prefix = indent * '  '
    for k in sorted(d):
        v = d[k]
        k = prefix + str(k)
        if isinstance(v, dict):
            print(k)
            print_data(v, indent + 1)
        else:
            if k.endswith('cent'):
                v = ' '.join(
                    str(tuple(j for j in i))
                    for i in v
                )
            elif isinstance(v, np.ndarray):
                v = str(v).replace('\n', '')
            print(k, '=', v)


if __name__ == '__main__':
    print_data(data)
