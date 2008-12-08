#!/usr/bin/env python
from __future__ import division, with_statement

__version__ = "$Revision$"

# Copyright 2008 Michael M. Hoffman <mmh1@washington.edu>

from collections import defaultdict
from contextlib import closing
from functools import partial
from gzip import open as _gzip_open
from os import extsep
import shutil
import sys
from tempfile import mkdtemp

from numpy import append, array, diff, empty, insert, intc
from path import path
from pkg_resources import resource_filename, resource_string
from tables import openFile, Filters, NoSuchNodeError

try:
    # Python 2.6
    PKG = __package__
except NameError:
    PKG = "segway"

PKG_DATA = ".".join([PKG, "data"])

FILTERS_GZIP = Filters(complevel=1)

EXT_GZ = "gz"
SUFFIX_GZ = extsep + EXT_GZ

DTYPE_IDENTIFY = intc
DTYPE_OBS_INT = intc
DTYPE_SEG_LEN = intc

data_filename = partial(resource_filename, PKG_DATA)
data_string = partial(resource_string, PKG_DATA)

# NamedTemporaryDir is based somewhat on Python 2.5.2
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

def walk_continuous_supercontigs(h5file):
    for supercontig in walk_supercontigs(h5file):
        try:
            yield supercontig, supercontig.continuous
        except NoSuchNodeError:
            continue

def fill_array(scalar, shape, dtype=None, *args, **kwargs):
    if dtype is None:
        dtype = array(scalar).dtype

    res = empty(shape, dtype, *args, **kwargs)
    res.fill(scalar)

    return res

def new_extrema(func, data, extrema):
    curr_extrema = func(data, 0)

    return func([extrema, curr_extrema], 0)

def init_num_obs(num_obs, continuous):
    curr_num_obs = continuous.shape[1]
    assert num_obs is None or num_obs == curr_num_obs

    return curr_num_obs

# XXX: suggest as default
def gzip_open(*args, **kwargs):
    return closing(_gzip_open(*args, **kwargs))

def get_tracknames(chromosome):
    return chromosome.root._v_attrs.tracknames.tolist()

def get_col_index(chromosome, trackname):
    return get_tracknames(chromosome).index(trackname)

def maybe_gzip_open(filename, *args, **kwargs):
    if filename.endswith(SUFFIX_GZ):
        return gzip_open(filename, *args, **kwargs)
    else:
        return open(filename, *args, **kwargs)

def load_coords(filename):
    if not filename:
        return

    coords = defaultdict(list)

    with maybe_gzip_open(filename) as infile:
        for line in infile:
            words = line.rstrip().split()
            chrom = words[0]
            start = int(words[1])
            end = int(words[2])

            coords[chrom].append((start, end))

    return dict((chrom, array(coords_list))
                for chrom, coords_list in coords.iteritems())

def get_chrom_coords(coords, chrom):
    """
    returns False if there are no coords on that chromosome
    returns None if there are no coords whatsoever
    """
    if coords:
        try:
            return coords[chrom]
        except KeyError:
            # nothing is included on that chromosome
            return array([])

def is_empty_array(arr):
    try:
        return arr.shape == (0,)
    except AttributeError:
        return False

def chrom_name(filename):
    return path(filename).namebase

def iter_chroms_coords(filenames, coords):
    for filename in filenames:
        print >>sys.stderr, filename
        chrom = chrom_name(filename)

        chr_include_coords = get_chrom_coords(coords, chrom)

        if is_empty_array(chr_include_coords):
            continue

        with openFile(filename) as chromosome:
            yield chrom, filename, chromosome, chr_include_coords

def find_segment_starts(data):
    """
    find segment starts and the data at those positions

    returns lists of len num_segments+1, num_segments
    """
    len_data = len(data)

    # unpack tuple, ignore rest
    end_pos, = diff(data).nonzero()

    # add one to get the start positions, and add a 0 at the beginning
    start_pos = insert(end_pos + 1, 0, 0)
    labels = data[start_pos]

    # after generating labels, add an extraneous start position so
    # where_seg+1 doesn't go out of bounds
    start_pos = append(start_pos, len_data)

    return start_pos, labels

def main(args=sys.argv[1:]):
    pass

if __name__ == "__main__":
    sys.exit(main())
