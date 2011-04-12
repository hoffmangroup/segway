#!/usr/bin/env python
from __future__ import division, with_statement

__version__ = "$Revision$"

# Copyright 2008-2009, 2011 Michael M. Hoffman <mmh1@washington.edu>

from collections import defaultdict
from contextlib import closing
from functools import partial
from gzip import open as _gzip_open
from itertools import repeat
from os import extsep
import shutil
import sys
from tempfile import mkdtemp

import colorbrewer
from numpy import (add, append, arcsinh, array, column_stack, diff, empty,
                   insert, intc, invert, isnan, maximum, zeros)

from optbuild import Mixin_UseFullProgPath, OptionBuilder_ShortOptWithSpace_TF
from path import path
from pkg_resources import resource_filename, resource_string
from tables import Filters, openFile

# XXX: check that these are all in use

# these are loaded by other modules indirectly
# ignore PyFlakes warnings here

try:
    # Python 2.6
    PKG = __package__
except NameError:
    PKG = "segway"

PKG_DATA = ".".join([PKG, "data"])

FILTERS_GZIP = Filters(complevel=1)

EXT_LIST = "list"
EXT_INT = "int"
EXT_FLOAT = "float32"
EXT_GZ = "gz"

SUFFIX_GZ = extsep + EXT_GZ

DTYPE_IDENTIFY = intc
DTYPE_OBS_INT = intc
DTYPE_SEG_LEN = intc

DISTRIBUTION_NORM = "norm"
DISTRIBUTION_GAMMA = "gamma"
DISTRIBUTION_ASINH_NORMAL = "asinh_norm"

# sentinel values
ISLAND_BASE_NA = 0
ISLAND_LST_NA = 0

ORD_A = ord("A")
ORD_C = ord("C")
ORD_G = ord("G")
ORD_T = ord("T")
ORD_a = ord("a")
ORD_c = ord("c")
ORD_g = ord("g")
ORD_t = ord("t")

data_filename = partial(resource_filename, PKG_DATA)
data_string = partial(resource_string, PKG_DATA)

NUM_COLORS = 8
SCHEME = colorbrewer.Dark2[NUM_COLORS]

KB = 2**10
MB = 2**20
GB = 2**30
TB = 2**40
PB = 2**50
EB = 2**60

OptionBuilder_GMTK = (Mixin_UseFullProgPath +
                      OptionBuilder_ShortOptWithSpace_TF)

VITERBI_PROG = OptionBuilder_GMTK("gmtkViterbi")

BED_SCORE = "1000"
BED_STRAND = "."

def extjoin(*args):
    return extsep.join(args)

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

def die(msg=""):
    if msg:
        print >>sys.stderr, msg
    sys.exit(1)

def get_col_index(chromosome, trackname):
    return get_tracknames(chromosome).index(trackname)

def get_label_color(label):
    color = SCHEME[label % NUM_COLORS]

    return ",".join(map(str, color))

# XXX: suggest as default
def fill_array(scalar, shape, dtype=None, *args, **kwargs):
    if dtype is None:
        dtype = array(scalar).dtype

    res = empty(shape, dtype, *args, **kwargs)
    res.fill(scalar)

    return res

# XXX: suggest as default
def gzip_open(*args, **kwargs):
    return closing(_gzip_open(*args, **kwargs))

def maybe_gzip_open(filename, mode="r", *args, **kwargs):
    if filename == "-":
        if mode.startswith("U"):
            raise NotImplementedError("U mode not implemented")
        elif mode.startswith("w") or mode.startswith("a"):
            return sys.stdout
        elif mode.startswith("r"):
            if "+" in mode:
                raise NotImplementedError("+ mode not implemented")
            else:
                return sys.stdin
        else:
            raise ValueError("mode string must begin with one of 'r', 'w', or 'a'")

    if filename.endswith(SUFFIX_GZ):
        return gzip_open(filename, mode, *args, **kwargs)

    return open(filename, mode, *args, **kwargs)

def constant(val):
    """
    constant values for defaultdict
    """
    return repeat(val).next

array_factory = constant(array([]))

# XXX: replace with genomedata.Chromosome.tracknames_continuous
def get_tracknames(chromosome):
    return chromosome.root._v_attrs.tracknames.tolist()

def init_num_obs(num_obs, continuous):
    curr_num_obs = continuous.shape[1]
    assert num_obs is None or num_obs == curr_num_obs

    return curr_num_obs

def new_extrema(func, data, extrema):
    curr_extrema = func(data, 0)

    return func([extrema, curr_extrema], 0)

# XXX: replace with iter(genomedata.Chromosome)
def walk_supercontigs(h5file):
    root = h5file.root

    for supercontig in h5file.walkGroups():
        if supercontig == root:
            continue

        yield supercontig

# XXX: replace with genomedata.Chromosome.itercontinuous
def walk_continuous_supercontigs(h5file):
    for supercontig in walk_supercontigs(h5file):
        try:
            yield supercontig, supercontig.continuous
        except NoSuchNodeError:
            continue

# XXX: duplicates bed.py?
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

    return defaultdict(array_factory, ((chrom, array(coords_list))
                       for chrom, coords_list in coords.iteritems()))

def get_chrom_coords(coords, chrom):
    """
    returns empty array if there are no coords on that chromosome
    returns None if there are no coords whatsoever
    """
    if coords:
        return coords[chrom]

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
    finds the start of each segment

    returns the start positions, and the labels at each position

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

def ceildiv(dividend, divisor):
    "integer ceiling division"

    # int(bool) means 0 -> 0, 1+ -> 1
    return (dividend // divisor) + int(bool(dividend % divisor))

def downsample_add(inarray, resolution):
    # downsample presence data into num_datapoints
    full_num_rows = inarray.shape[0]

    downsampled_num_rows = ceildiv(full_num_rows, resolution)
    remainder = full_num_rows % resolution
    downsampled_shape = [downsampled_num_rows] + list(inarray.shape[1:])

    res = zeros(downsampled_shape, inarray.dtype)

    # if there's no remainder, then only use loop 0
    if remainder == 0:
        remainder = resolution

    # loop 0: every bit
    for index in xrange(remainder):
        res += inarray[index::resolution]

    # loop 1: remainder
    for index in xrange(remainder, resolution):
        res[:-1] += inarray[index::resolution]

    return res

def make_dinucleotide_int_data(seq):
    """
    makes an array with two columns, one with 0..15=AA..TT and the other
    as a presence variable. Set column one to 0 when not present
    """
    nucleotide_int_data = (((seq == ORD_A) + (seq == ORD_a))
                           + ((seq == ORD_C) + (seq == ORD_c)) * 2
                           + ((seq == ORD_G) + (seq == ORD_g)) * 3
                           + ((seq == ORD_T) + (seq == ORD_t)) * 4) - 1
    nucleotide_missing = nucleotide_int_data == -1

    # rewrite all Ns as A now that you have the missingness mask
    nucleotide_int_data[nucleotide_int_data == -1] = 0
    col_shape = (len(nucleotide_int_data)-1,)

    # first column: dinucleotide: AA..TT=0..15
    # combine, and add extra AA stub at end, which will be set missing
    # 0 AA AC AG AT
    # 4 CA CC CG CT
    # 8 GA GC GG GT
    # 12 TA TC TG TT
    dinucleotide_int_data = empty(col_shape, DTYPE_OBS_INT)
    add(nucleotide_int_data[:-1] * 4, nucleotide_int_data[1:],
        dinucleotide_int_data)

    # second column: presence_dinucleotide: some missing=0; all present = 1
    # there are so few N boundaries that it is okay to
    # disregard the whole dinucleotide when half is N
    dinucleotide_missing = (nucleotide_missing[:-1] + nucleotide_missing[1:])

    dinucleotide_presence = empty(dinucleotide_missing.shape, DTYPE_OBS_INT)
    invert(dinucleotide_missing, dinucleotide_presence)

    # XXXopt: set these up properly in the first place instead of
    # column_stacking at the end
    return column_stack([dinucleotide_int_data, dinucleotide_presence])

def _save_observations_window(float_filename, int_filename, float_data,
                             resolution, distribution, seq_data=None,
                             supervision_data=None):
    # this is now a naked function so that it can be called by task.py

    # input function in GMTK_ObservationMatrix.cc:
    # ObservationMatrix::readBinSentence

    # input per frame is a series of float32s, followed by a series of
    # int32s it is better to optimize both sides here by sticking all
    # the floats in one file, and the ints in another one
    int_blocks = []
    if float_data is not None:
        if distribution == DISTRIBUTION_ASINH_NORMAL:
            float_data = arcsinh(float_data)

        mask_missing = isnan(float_data)

        # output -> presence_data -> int_blocks
        # done in two steps so I can specify output type
        presence_data = empty(mask_missing.shape, DTYPE_OBS_INT)
        invert(mask_missing, presence_data)

        num_datapoints = downsample_add(presence_data, resolution)

        int_blocks.append(num_datapoints)

        # so that there is no divide by zero
        num_datapoints_min_1 = maximum(num_datapoints, 1)

        # make float
        float_data[mask_missing] = 0.0

        float_data_downsampled = downsample_add(float_data, resolution)
        float_data_downsampled /= num_datapoints_min_1
        float_data_downsampled.tofile(float_filename)

    if seq_data is not None:
        assert resolution == 1 # not implemented yet
        int_blocks.append(make_dinucleotide_int_data(seq_data))

    if supervision_data is not None:
        assert resolution == 1 # not implemented yet
        int_blocks.append(supervision_data)

    # XXXopt: use the correctly sized matrix in the first place
    int_data = column_stack(int_blocks)
    int_data.tofile(int_filename)

# XXX: there is a new genomedata interface that should be able to
# replace the callers to this
def _make_continuous_cells(supercontig, start, end, track_indexes):
    continuous = supercontig.continuous
    if continuous is None:
        return

    # window_start: relative to the beginning of the supercontig
    window_start = start - supercontig.start
    window_end = end - supercontig.start

    # XXXopt: reading all the extra tracks is probably quite wasteful
    # given the genomedata striping pattern; it is probably better to
    # read one at a time and stick into an array
    min_col = track_indexes.min()
    max_col = track_indexes.max() + 1

    # first, extract a contiguous subset of the tracks in the dataset,
    # which is a superset of the tracks that are used
    rows = continuous[window_start:window_end, min_col:max_col]

    # extract only the tracks that are used correct for min_col offset
    return rows[..., track_indexes - min_col]

def make_filelistpath(dirpath, ext):
    return dirpath / extjoin(ext, EXT_LIST)

def main(args=sys.argv[1:]):
    pass

if __name__ == "__main__":
    sys.exit(main())
