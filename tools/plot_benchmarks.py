# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2021 Scipp contributors (https://github.com/scipp)
"""
Plot the results of benchmarks.

The script parses the CSV files generated by Google Benchmark with the options
``--benchmark_out_format=csv --benchmark_out=<path>.csv
  --benchmark_repetitions=<n>``.
"""

import argparse
import math
from pathlib import Path
import re

import matplotlib.pyplot as plt
import pandas as pd

# Regex to identify the header of the table containing the actual data.
TABLE_HEADER_PATTERN = re.compile(r'^name,iterations,.*')
# Regex to turn full names (BM_xxx/yyy_zzz) into base names (xxx)
# and 'aggregates' (zzz).
NAME_PATTERN = re.compile(
    r'(BM_)?(?P<name>[^/]+)(/[^_]*)?(_(?P<aggregate>\w+))?')

# Table columns written by Google Benchmark by default.
DEFAULT_COLUMN_NAMES = [
    'name', 'iterations', 'real_time', 'cpu_time', 'time_unit',
    'bytes_per_second', 'items_per_second', 'label'
]


def seek_table(file_handle):
    """Advance file_handle to the beginning of the table."""
    line = ''
    pos = 0
    while not TABLE_HEADER_PATTERN.match(line):
        pos = file_handle.tell()
        line = file_handle.readline()
    file_handle.seek(pos)


def process_errors(dataframe):
    """Search for errors and remove the error columns."""
    # if there was no error => error_occurred = NaN
    errors = dataframe.query('error_occurred == error_occurred')
    del dataframe['error_occurred']
    del dataframe['error_message']
    if not errors.empty:
        print('There are errors in the input files:')
        print(errors)
        raise RuntimeError('Input files contain errors')


def preprocess(dataframe, name_filter):
    """Prepare a DataFrame for plotting."""

    dataframe = dataframe[dataframe.name.str.contains(name_filter)]
    if dataframe.empty:
        raise ValueError(f'No benchmarks with name {name_filter.pattern}')
    dataframe['aggregate'] = dataframe.name.map(
        lambda full_name: NAME_PATTERN.match(full_name)['aggregate'])
    dataframe.name = dataframe.name.map(
        lambda full_name: NAME_PATTERN.match(full_name)['name'])

    process_errors(dataframe)

    time_units = dataframe['time_unit'].unique()
    if time_units != ['ns']:
        raise NotImplementedError(f'Unsupported time units: {time_units}')

    return dataframe


def load_file(fname):
    """Load a single file with benchmark results."""
    assert fname.suffix == '.csv'
    with open(fname, 'r') as f:
        seek_table(f)
        data = pd.read_csv(f)
    data['file'] = fname
    return data


def load_data(fnames, name_filter):
    """Load multiple files and merge their contents."""
    return preprocess(pd.concat([load_file(fname) for fname in fnames]),
                      name_filter)


def designate_columns(data, plot_dims, ignored):
    """Return a list with roles for each column in ``data``."""
    return [
        'stack' if c == 'file' else
        'aggregate' if c == 'aggregate' else 'plot_dim' if c in plot_dims else
        'ignored' if c in ignored or c in DEFAULT_COLUMN_NAMES else 'meta'
        for c in data.columns
    ]


def group_plots(data, designations):
    """Return the number of distinct plot groups and the groups themselves."""
    meta_columns = [
        c for c, d in zip(data.columns, designations) if d == 'meta'
    ]
    groups = data.groupby(meta_columns)
    return len(groups), [group for _, group in groups]


def plot_title(data, designations):
    """Format a title for a plot of ``data``."""
    return ', '.join(f'{c}={data[c].unique()[0]}'
                     for c, d in zip(data.columns, designations)
                     if d == 'meta')


def get_unit(data, name):
    """Return the unit for an axis label for column ``name``."""
    if name in ('real_time', 'cpu_time'):
        return f' [{data.time_unit.unique()[0]}]'
    return ''


def plot(data, xname, yname, ignored, xscale='log'):
    """Plot the data."""

    designations = designate_columns(data, (xname, yname), ignored)
    n_subplots, groups = group_plots(data, designations)
    if n_subplots == 0:
        return

    fig = plt.figure()
    fig.suptitle(data.name.unique()[0])
    nx = min(3, n_subplots)
    ny = math.ceil(n_subplots / nx)
    for igroup, group in enumerate(groups):
        ax = fig.add_subplot(nx, ny, igroup + 1)
        ax.set_title(plot_title(group, designations))
        ax.set_xlabel(xname + get_unit(data, xname))
        ax.set_ylabel(yname + get_unit(data, yname))
        ax.set_xscale(xscale)

        for iline, (fname, line) in enumerate(group.groupby('file')):
            mean = line.query('aggregate == "mean"')
            median = line.query('aggregate == "median"')
            # TODO custom counters are all 0 in the stddev row
            #  => can't use it here
            # stddev = line.query('aggregate == "stddev"')

            ax.plot(mean[xname].to_numpy(),
                    mean[yname].to_numpy(),
                    marker='.',
                    c=f'C{iline}',
                    label=fname)
            ax.plot(median[xname].to_numpy(),
                    median[yname].to_numpy(),
                    marker='_',
                    ls='',
                    c=f'C{iline}')

        if igroup == 0:
            ax.legend()
    fig.tight_layout()


def make_name_filter(names):
    """Return a regex that filters out the given name(s)."""
    name_pattern = NAME_PATTERN.match(names)['name']
    return re.compile(name_pattern)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument('infile', nargs='+', type=Path, help='Input files')
    parser.add_argument('-n',
                        '--names',
                        type=make_name_filter,
                        help='Filter for benchmark names (regex)',
                        metavar='filter',
                        default='.*')
    parser.add_argument('-x',
                        '--xaxis',
                        default='size',
                        help='Quantity to display on the x-axis')
    parser.add_argument('-y',
                        '--yaxis',
                        default='real_time',
                        help='Quantity to display on the y-axis')
    parser.add_argument('--xscale', default='log',
                        help='Use a linear scale on the x-axis')
    parser.add_argument(
        '--ignore',
        type=lambda s: s.split(','),
        default='',
        help='Quantities to ignore when splitting benchmarks into groups.')
    return parser.parse_args()


def main():
    args = parse_args()
    data = load_data(args.infile, args.names)
    for _, benchmark_data in data.groupby('name'):
        plot(benchmark_data, args.xaxis, args.yaxis,
             args.ignore, xscale=args.xscale)
    plt.show()


if __name__ == '__main__':
    main()
