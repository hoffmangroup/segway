#!/usr/bin/env python
from __future__ import division, with_statement

"""
h5histogram: prints histogram
"""

__version__ = "$Revision$"

# Copyright 2008 Michael M. Hoffman <mmh1@washington.edu>

import sys

from functools import partial
from numpy import array, histogram, isfinite, NINF, PINF
from tables import openFile

from ._util import (get_col_index as _get_col_index, iter_chroms_coords,
                    load_coords, walk_continuous_supercontigs)

BINS = 100
FIELDNAMES = ["lower_edge", "count"]

def get_col_index(chromosome, trackname):
    if trackname is None:
        return 0
    else:
        return _get_col_index(chromosome, trackname)

def calc_range(trackname, filenames):
    # not limited to include_coords, so scale is same
    minimum = PINF
    maximum = NINF

    for filename in filenames:
        with openFile(filename) as chromosome:
            col_index = get_col_index(chromosome, trackname)

            attrs = chromosome.root._v_attrs
            minimum = min(minimum, attrs.mins[col_index])
            maximum = max(maximum, attrs.maxs[col_index])

    return minimum, maximum

def calc_histogram(trackname, filenames, data_range, include_coords):
    histogram_custom = partial(histogram, bins=BINS, range=data_range,
                               new=True)

    hist, edges = histogram_custom(array([]))
    chrom_iterator = iter_chroms_coords(filenames, include_coords)
    for chrom, filename, chr_include_coords in chrom_iterator:
        with openFile(filename) as chromosome:
            col_index = get_col_index(chromosome, trackname)

            supercontig_walker = walk_continuous_supercontigs(chromosome)
            for supercontig, continuous in supercontig_walker:
                supercontig_attrs = supercontig._v_attrs
                supercontig_start = supercontig_attrs.start

                if include_coords:
                    # adjust coords
                    chr_include_coords = chr_include_coords - supercontig_start

                    # set all negative coords to 0
                    chr_include_coords[chr_include_coords < 0] = 0
                else:
                    # slice(None, None) means the whole sequence
                    chr_include_coords = [[None, None]]

                for coords in chr_include_coords:
                    row_slice = slice(*coords)

                    col = continuous[row_slice, col_index]
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

def h5histogram(trackname, filenames, include_coords_filename=None):
    print "\t".join(FIELDNAMES)

    include_coords = load_coords(include_coords_filename)

    # two passes to avoid running out of memory
    data_range = calc_range(trackname, filenames)
    hist, edges = calc_histogram(trackname, filenames, data_range,
                                 include_coords)

    print_histogram(hist, edges)

def parse_options(args):
    from optparse import OptionParser

    usage = "%prog [OPTION]... FILE..."
    version = "%%prog %s" % __version__
    parser = OptionParser(usage=usage, version=version)
    # this is a 0-based file (I know because ENm008 starts at position 0)
    parser.add_option("--include-coords", metavar="FILE",
                      help="limit to genomic coordinates in FILE")

    parser.add_option("-c", "--col", metavar="COL",
                      help="write values in column COL (default first column)")

    options, args = parser.parse_args(args)

    if not len(args) >= 1:
        parser.print_usage()
        sys.exit(1)

    return options, args

def main(args=sys.argv[1:]):
    options, args = parse_options(args)

    return h5histogram(options.col, args, options.include_coords)

if __name__ == "__main__":
    sys.exit(main())
