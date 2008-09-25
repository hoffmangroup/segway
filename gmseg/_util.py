#!/usr/bin/env python
from __future__ import division

__version__ = "$Revision$"

# Copyright 2008 Michael M. Hoffman <mmh1@u.washington.edu>

from functools import partial
import shutil
import sys
from tempfile import mkdtemp

from numpy import array, empty, NINF, PINF
from pkg_resources import resource_filename, resource_string

DATA_PKG = "gmseg.data"

data_filename = partial(resource_filename, DATA_PKG)
data_string = partial(resource_string, DATA_PKG)

# NamedTemporaryFile is based somewhat on Python 2.5.2
# tempfile._TemporaryFileWrapper
#
# Original Copyright (c) 2001, 2002, 2003, 2004, 2005, 2006 Python
# Software Foundation; All Rights Reserved
#
# License at http://www.python.org/download/releases/2.5.2/license/

# XXX: submit for inclusion in core

class NamedTemporaryDir(object):
    def __init__(self, *args, **kwargs):
        self.name = mkdtemp(*args, **kwargs)
        self.close_called = False
        self.rmtree = shutil.rmtree # want a function, not an unbound method

    def __enter__(self):
        return self

    def close(self):
        if not self.close_called:
            self.close_called = True
            self.rmtree(self.name)

    def __del__(self):
        self.close()

    def __exit__(self, exc, value, tb):
        self.close()

class LightIterator(object):
    def __init__(self, handle):
        self._handle = handle
        self._defline = None

    def __iter__(self):
        return self

    def next(self):
        lines = []
        defline_old = self._defline

        while 1:
            line = self._handle.readline()
            if not line:
                if not defline_old and not lines:
                    raise StopIteration
                if defline_old:
                    self._defline = None
                    break
            elif line[0] == '>':
                self._defline = line[1:].rstrip()
                if defline_old or lines:
                    break
                else:
                    defline_old = self._defline
            else:
                lines.append(line.rstrip())

        return defline_old, ''.join(lines)

def walk_supercontigs(h5file):
    root = h5file.root

    for supercontig in h5file.walkGroups():
        if supercontig == root:
            continue

        yield supercontig

def fill_array(scalar, shape, dtype=None, *args, **kwargs):
    if dtype is None:
        dtype = array(scalar).dtype

    res = empty(shape, dtype, *args, **kwargs)
    res.fill(scalar)

    return res


def new_extrema(func, data, extrema):
    curr_extrema = func(data, 0)

    return func([extrema, curr_extrema], 0)

# because this would otherwise be duplicative
def init_mins_maxs(num_obs, mins, maxs, continuous):
    curr_num_obs = continuous.shape[1]
    if num_obs is None:
        ## setup at first array
        num_obs = curr_num_obs

        extrema_shape = (num_obs,)
        mins = fill_array(PINF, extrema_shape)
        maxs = fill_array(NINF, extrema_shape)
    else:
        ## ensure homogeneity
        assert num_obs == curr_num_obs

    return num_obs, mins, maxs

def main(args=sys.argv[1:]):
    pass

if __name__ == "__main__":
    sys.exit(main())
