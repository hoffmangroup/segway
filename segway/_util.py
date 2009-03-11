#!/usr/bin/env python
from __future__ import division, with_statement

__version__ = "$Revision$"

# Copyright 2008-2009 Michael M. Hoffman <mmh1@washington.edu>

from collections import defaultdict
from contextlib import contextmanager
from functools import partial
from os import extsep
import shutil
import sys
from tempfile import mkdtemp

from DRMAA import Session as _Session
from numpy import append, array, diff, insert, intc
from path import path
from pkg_resources import resource_filename, resource_string
from tables import openFile

# these are loaded by other modules indirectly
# XXX: check that they are all in use
from genomedata._util import (EXT_GZ, fill_array, FILTERS_GZIP, get_tracknames,
                              gzip_open, init_num_obs, LightIterator,
                              new_extrema, walk_supercontigs,
                              walk_continuous_supercontigs)

try:
    # Python 2.6
    PKG = __package__
except NameError:
    PKG = "segway"

DRMSINFO_PREFIX = "GE" # XXX: only SGE is supported for now

PKG_DATA = ".".join([PKG, "data"])

SUFFIX_GZ = extsep + EXT_GZ

DTYPE_IDENTIFY = intc
DTYPE_OBS_INT = intc
DTYPE_SEG_LEN = intc

# sentinel values
ISLAND_BASE_NA = 0
ISLAND_LST_NA = 0

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
# XXX: is this still in use?
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

# XXX: suggest upstream as addition to DRMAA-python
@contextmanager
def Session(*args, **kwargs):
    res = _Session()
    res.init(*args, **kwargs)

    assert res.DRMSInfo.startswith(DRMSINFO_PREFIX)

    try:
        yield res
    finally:
        res.exit()

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
    returns empty array if there are no coords on that chromosome
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

# XXX: replace with stuff from prep_observations()
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
