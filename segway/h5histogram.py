#!/usr/bin/env python
from __future__ import division, with_statement

"""
h5histogram: prints histogram

XXX a lot of this is copied and updated in the external validation tool
"""

__version__ = "$Revision$"

# Copyright 2008 Michael M. Hoffman <mmh1@washington.edu>

import sys

from collections import defaultdict
from functools import partial
from numpy import (array, concatenate, histogram, iinfo, isfinite, ndarray,
                   NINF, PINF, zeros)
from tables import openFile

from ._util import (DTYPE_IDENTIFY, fill_array,
                    get_col_index as _get_col_index, iter_chroms_coords,
                    load_coords, walk_continuous_supercontigs)

FIELDNAMES = ["lower_edge", "count"]

IINFO_IDENTIFY = iinfo(DTYPE_IDENTIFY)
MAX_IDENTIFY = IINFO_IDENTIFY.max # sentinel for padding

def get_col_index(chromosome, trackname):
    if trackname is None:
        return 0
    else:
        return _get_col_index(chromosome, trackname)

def calc_range(trackname, filenames):
    # not limited to include_coords, so scale is always the same
    minimum = PINF
    maximum = NINF

    for filename in filenames:
        with openFile(filename) as chromosome:
            col_index = get_col_index(chromosome, trackname)

            attrs = chromosome.root._v_attrs
            minimum = min(minimum, attrs.mins[col_index])
            maximum = max(maximum, attrs.maxs[col_index])

    return minimum, maximum

def calc_histogram(trackname, filenames, data_range, num_bins, include_coords,
                   include_identify_dict, identify_label):
    histogram_custom = partial(histogram, bins=num_bins, range=data_range,
                               new=True)

    hist, edges = histogram_custom(array([]))
    chrom_iterator = iter_chroms_coords(filenames, include_coords)
    for chrom, filename, chromosome, chr_include_coords in chrom_iterator:
        assert not chromosome.root._v_attrs.dirty

        col_index = get_col_index(chromosome, trackname)

        supercontig_walker = walk_continuous_supercontigs(chromosome)
        for supercontig, continuous in supercontig_walker:
            supercontig_attrs = supercontig._v_attrs
            supercontig_start = supercontig_attrs.start
            supercontig_end  = supercontig_attrs.end

            if include_coords:
                # adjust coords
                supercontig_include_coords = (chr_include_coords
                                              - supercontig_start)

                # set all negative coords to 0
                supercontig_include_coords[supercontig_include_coords < 0] = 0
            else:
                # slice(None, None) means the whole sequence
                supercontig_include_coords = [[None, None]]

            for coords in supercontig_include_coords:
                if coords[0] >= len(continuous):
                    continue

                row_slice = slice(*coords)

                col = continuous[row_slice, col_index]

                if include_identify_dict:
                    include_identify_chunks = include_identify_dict[chrom]

                    col_bitmap = zeros(col.shape, bool)

                    coords_start, coords_end = coords
                    if coords_start is None:
                        coords_start = 0
                    if coords_end is None:
                        coords_end = supercontig_end

                    for chunk in include_identify_chunks:
                        chunk_attrs = chunk.root._v_attrs
                        chunk_start = chunk_attrs.start - supercontig_start
                        chunk_end = chunk_attrs.end - supercontig_start

                        chunk_identify = chunk.root.identify
                        assert chunk_identify.atom.dtype == DTYPE_IDENTIFY

                        if (chunk_start >= coords_end
                            or chunk_end <= coords_start):
                            continue

                        chunk_identify_array = None

                        chunk_start_offset = coords_start - chunk_start
                        if chunk_start_offset > 0:
                            chunk_identify_array = \
                                chunk_identify[chunk_start_offset:]
                        elif chunk_start_offset < 0:
                            padding_shape = (-chunk_start_offset,)
                            padding = fill_array(MAX_IDENTIFY, padding_shape,
                                                 DTYPE_IDENTIFY)
                            padded_list = [padding, chunk_identify.read()]

                            chunk_identify_array = concatenate(padded_list)

                        # if there is not enough padding at end,
                        # things will work. Need to correct if there
                        # is too much
                        if chunk_end > coords_end:
                            # -1 if it is one longer, etc.
                            chunk_end_offset = chunk_end-coords_end
                            chunk_identify_array = \
                                chunk_identify[:chunk_end_offset]

                        if chunk_identify_array is None:
                            chunk_identify_array = chunk_identify.read()

                        assert isinstance(chunk_identify_array, ndarray)

                        assert (len(chunk_identify_array)
                                <= coords_end - coords_start)

                        loc_true = chunk_identify_array == identify_label
                        col_bitmap[loc_true] = True

                    col = col[col_bitmap]

                col_finite = col[isfinite(col)]

                # if it has at least one row (it isn't truncated
                # away by the include_coords)
                if col_finite.shape[0]:
                    if coords[0] is not None:
                        coords_tuple = tuple(coords + supercontig_start)
                        print >>sys.stderr, " (%s, %s)" % coords_tuple

                    hist_supercontig, edges_supercontig = \
                        histogram_custom(col_finite)

                    assert (edges_supercontig == edges).all()
                    hist += hist_supercontig

    return hist, edges

def print_histogram(hist, edges):
    for row in zip(edges, hist.tolist() + ["NA"]):
        print "\t".join(map(str, row))

def load_include_identify(filelistname):
    if filelistname is None:
        return {}

    res = defaultdict(list)

    with open(filelistname) as filelist:
        for line in filelist:
            filename = line.rstrip()

            # XXX: these never get closed
            identify = openFile(filename)
            chrom = identify.root._v_attrs.chrom
            res[chrom].append(identify)

    return res

def h5histogram(trackname, filenames, num_bins, include_coords_filename=None,
                include_identify_filelistname=None, identify_label=1):
    print "\t".join(FIELDNAMES)

    # two passes to avoid running out of memory
    data_range = calc_range(trackname, filenames)

    include_coords = load_coords(include_coords_filename)
    include_identify_dict = \
        load_include_identify(include_identify_filelistname)

    try:
        hist, edges = calc_histogram(trackname, filenames, data_range,
                                     num_bins, include_coords,
                                     include_identify_dict, identify_label)
    finally:
        for include_identify_h5files in include_identify_dict.itervalues():
            for include_identify_h5file in include_identify_h5files:
                include_identify_h5file.close()

    print_histogram(hist, edges)

def parse_options(args):
    from optparse import OptionParser

    usage = "%prog [OPTION]... FILE..."
    version = "%%prog %s" % __version__
    parser = OptionParser(usage=usage, version=version)
    # this is a 0-based file (I know because ENm008 starts at position 0)
    parser.add_option("--include-coords", metavar="FILE",
                      help="limit to genomic coordinates in FILE")

    parser.add_option("--include-identify", metavar="FILELIST",
                      help="limit to label identified in files in FILELIST")

    parser.add_option("--identify-label", metavar="LABEL", default=1, type=int,
                      help="limit to LABEL in a list of specified identify"
                      " files")

    parser.add_option("-c", "--col", metavar="COL",
                      help="write values in column COL (default first column)")

    parser.add_option("-b", "--num-bins", metavar="BINS", type=int,
                      default=100, help="use BINS bins")

    options, args = parser.parse_args(args)

    if not len(args) >= 1:
        parser.print_usage()
        sys.exit(1)

    return options, args

def main(args=sys.argv[1:]):
    options, args = parse_options(args)

    return h5histogram(options.col, args, options.num_bins,
                       options.include_coords, options.include_identify,
                       options.identify_label)

if __name__ == "__main__":
    sys.exit(main())
